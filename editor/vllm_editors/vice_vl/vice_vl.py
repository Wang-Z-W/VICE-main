"""
VICE (Sequential-edit memory) for VLLMs.

Maintains persistent sequence memory during continual edits:
- Append each new fact and its context to memory using sentence embeddings
- Retrieve similar memory items at inference time via semantic search
- Concatenate retrieved context before decoding

Reproduces the exact algorithm from ComprehendEdit VICE.py with:
- KMeans clustering for diverse dataset selection (ComprehendEdit.py:254)
- Embedding-based retrieval (ike_main.py:250)
- Automatic inference context augmentation via prompts_imgs_target_to_xym wrapping
"""

import json
import os
import re
from ...vllms_for_edit.base import BaseVLLMForEdit
from ..base import VLLMBaseEditor
from ...base import BaseConfig
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util
import torch
from PIL import Image


@dataclass
class VICEvlConfig(BaseConfig):
    edit_model_name: str
    sentence_model_name: str = "all-MiniLM-L6-v2"

class VICEvl(VLLMBaseEditor):
    """Sequential-edit extension for continual model editing with embedding-based memory."""

    _SUMMARY_FIELD_PATTERNS = {
        "target": re.compile(
            r"(?is)((?:^|\n)\s*Target Evidence\s*:\s*)(.*?)(?=\n\s*(?:Distractors|Contrast)\s*:|\Z)"
        ),
        "distractors": re.compile(
            r"(?is)((?:^|\n)\s*Distractors\s*:\s*)(.*?)(?=\n\s*Contrast\s*:|\Z)"
        ),
    }
    _SUMMARY_BLOCK_PATTERN = re.compile(r"(?is)(Image Summary:\s*)(.*?)(\nNew Fact:)")
    _SUMMARY_ABLATION_ALIASES = {
        None: "none",
        "": "none",
        "none": "none",
        "full": "none",
        "no_target": "target_empty",
        "target_empty": "target_empty",
        "remove_target": "target_empty",
        "no_evidence": "target_empty",
        "evidence_empty": "target_empty",
        "no_distractors": "distractors_empty",
        "distractors_empty": "distractors_empty",
        "remove_distractors": "distractors_empty",
        "no_distractor": "distractors_empty",
        "distractor_empty": "distractors_empty",
        "swap_fields": "swap_fields",
        "field_swap": "swap_fields",
        "swap_target_distractors": "swap_fields",
        "target_distractor_swap": "swap_fields",
    }

    def __init__(
        self,
        vllm: BaseVLLMForEdit,
        config: VICEvlConfig,
        device: list,
        sim_threshold: float,
        use_image_summary: bool = True,
        image_summary_ablation: str = "none",
    ):
        super().__init__(vllm, device)
        self.cfg = config
        self.sim_threshold = sim_threshold          # min cosine-sim to accept a retrieval
        self.use_image_summary = use_image_summary
        self.image_summary_ablation = self._normalize_summary_ablation(image_summary_ablation)
        self.image_rerank_candidates = 10           # top-k candidates to rerank by image similarity
        self.img_sum_device = self.device[0]        # device for CLIP (image summary + scope encoding)
        self.sequence_memory = self._empty_sequence_memory()  # {new_facts, contexts, embeddings, scope_embeddings, img_embeddings}
        self.stored_example_bundles = None           # diverse ICL dataset (list of edit bundles)
        self.stored_sentences = None                 # diverse ICL: sentence strings
        self.stored_embeddings = None                # diverse ICL: sentence-transformer embeddings
        self.stored_text_only_sentences = None       # diverse ICL (text-only fallback): sentence strings
        self.stored_text_only_embeddings = None      # diverse ICL (text-only fallback): embeddings
        self._scope_embedding_cache = {}             # (prompt, image_key) -> projected scope embedding
        self._image_embedding_cache = {}             # image_path -> CLIP image embedding
        self.scope_projection_head = None            # MLP: CLIP concat feat -> L2-normed scope embedding
        self.icl_role_order = [
            "reliability",
            "image_rephrase",
            "text_rephrase",
            "locality_text",
            "locality_image",
        ]

        # Initialize sentence transformer for embedding
        self.sentence_model = SentenceTransformer(config.sentence_model_name, device=self.device[0])

        # Image summary cache: loaded from preprocess_image_summaries.py output
        self._image_summary_cache = {}

        # Wrap vllm.prompts_imgs_target_to_xym() to include sequence memory context
        self._orig_prompts_imgs_target_to_xym = self.vllm.prompts_imgs_target_to_xym
        self.vllm.prompts_imgs_target_to_xym = self._augmented_prompts_imgs_target_to_xym

        if self.image_summary_ablation != "none":
            print(f"VICE image summary ablation: {self.image_summary_ablation}")

    def name_of_editor_and_model(self) -> Tuple[str, str]:
        return 'vice_vl', self.cfg.edit_model_name

    def if_can_batch_edit(self) -> bool:
        return False

    def _empty_sequence_memory(self) -> Dict[str, List]:
        return {
            "new_facts": [],
            "contexts": [],
            "fact_sentences": [],
            "embeddings": [],       # text embeddings from sentence-transformer
            "scope_embeddings": [],  # learned CLIP scope embeddings
            "img_embeddings": [],   # image embeddings from CLIP
        }

    def _normalize_summary_ablation(self, mode: Optional[str]) -> str:
        key = None if mode is None else str(mode).strip().lower().replace("-", "_")
        if key not in self._SUMMARY_ABLATION_ALIASES:
            valid = sorted(k for k in self._SUMMARY_ABLATION_ALIASES if isinstance(k, str) and k)
            raise ValueError(f"Unsupported image_summary_ablation={mode}. Valid values: {valid}")
        return self._SUMMARY_ABLATION_ALIASES[key]

    def _apply_image_summary_ablation(self, summary: str) -> str:
        if self.image_summary_ablation == "none" or not summary:
            return summary

        if self.image_summary_ablation == "swap_fields":
            target_match = self._SUMMARY_FIELD_PATTERNS["target"].search(summary)
            distractor_match = self._SUMMARY_FIELD_PATTERNS["distractors"].search(summary)
            if target_match is None or distractor_match is None:
                return summary

            target_text = target_match.group(2).strip()
            distractor_text = distractor_match.group(2).strip()
            swapped = self._SUMMARY_FIELD_PATTERNS["target"].sub(
                lambda match: f"{match.group(1)}{distractor_text}",
                summary,
                count=1,
            )
            swapped = self._SUMMARY_FIELD_PATTERNS["distractors"].sub(
                lambda match: f"{match.group(1)}{target_text}",
                swapped,
                count=1,
            )
            return swapped

        field_name = "target" if self.image_summary_ablation == "target_empty" else "distractors"
        pattern = self._SUMMARY_FIELD_PATTERNS[field_name]
        ablated, count = pattern.subn(lambda match: f"{match.group(1)}EMPTY", summary, count=1)
        return ablated if count else summary

    def _apply_image_summary_ablation_to_sentence(self, sentence: str) -> str:
        if self.image_summary_ablation == "none" or "Image Summary:" not in sentence:
            return sentence

        def replace_block(match):
            return f"{match.group(1)}{self._apply_image_summary_ablation(match.group(2))}{match.group(3)}"

        return self._SUMMARY_BLOCK_PATTERN.sub(replace_block, sentence, count=1)

    def _apply_image_summary_ablation_to_obj(self, obj):
        if isinstance(obj, str):
            return self._apply_image_summary_ablation_to_sentence(obj)
        if isinstance(obj, list):
            return [self._apply_image_summary_ablation_to_obj(item) for item in obj]
        if isinstance(obj, dict):
            return {key: self._apply_image_summary_ablation_to_obj(value) for key, value in obj.items()}
        return obj

    def _get_clip_model(self):
        """Lazily load CLIP for image embeddings (reused across all calls)."""
        if not hasattr(self, '_clip_model'):
            from transformers import CLIPModel, CLIPProcessor
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.img_sum_device)
            self._clip_model.eval()
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return self._clip_model, self._clip_processor

    def _encode_image(self, image, image_path: str = None) -> torch.Tensor:
        """Encode a single image to L2-normalized CLIP embedding [1, d]."""
        cache_key = image_path if image_path else (image if isinstance(image, str) else id(image))
        cached = self._image_embedding_cache.get(cache_key)
        if cached is not None:
            return cached.to(self.img_sum_device)

        model, processor = self._get_clip_model()
        if isinstance(image, torch.Tensor) and image.dim() == 4:
            image = image.squeeze(0)
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(self.img_sum_device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1)
        self._image_embedding_cache[cache_key] = emb.cpu()
        return emb

    def restore_to_original_model(self):
        """Reset to original model state."""
        self.sequence_memory = self._empty_sequence_memory()
        self._scope_embedding_cache = {}
        self._image_embedding_cache = {}

    def _encode_scope(self, prompt: str, image, image_path: str = None) -> Optional[torch.Tensor]:
        """Encode prompt+image to scope embedding via CLIP + projection head."""
        if self.scope_projection_head is None:
            return None

        img_key = image_path if image_path else (image if isinstance(image, str) else id(image))
        cache_key = (prompt, img_key)
        cached = self._scope_embedding_cache.get(cache_key)
        if cached is not None:
            return cached.to(self.device[0])

        model, processor = self._get_clip_model()
        text_inputs = processor(text=[prompt], padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.img_sum_device)
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        if image is None:
            image_features = torch.zeros_like(text_features)
        else:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            image_inputs = processor(images=[image], return_tensors="pt").to(self.img_sum_device)
            with torch.no_grad():
                image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        scope_feat = torch.cat([text_features, image_features], dim=-1).to(self.device[0])
        with torch.no_grad():
            emb = self.scope_projection_head(scope_feat)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        self._scope_embedding_cache[cache_key] = emb.cpu()
        return emb

    def load_classifier(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        projection = torch.nn.Sequential(
            torch.nn.Linear(ckpt["feature_dim"], ckpt["feature_dim"]),
            torch.nn.ReLU(),
            torch.nn.Linear(ckpt["feature_dim"], ckpt["projection_dim"]),
        ).to(self.device[0])
        projection.load_state_dict(ckpt["projection_state_dict"])
        projection.eval()
        self.scope_projection_head = projection
        self.scope_classifier_temperature = ckpt.get("temperature", 0.07)
        self.scope_classifier_feature_dim = ckpt["feature_dim"]
        self.scope_classifier_projection_dim = ckpt["projection_dim"]
        self._scope_embedding_cache = {}
        print(f"Loaded VICE scope classifier from {ckpt_path}")

    def _bundle_to_text_only_sentences(self, bundle: Dict) -> List[str]:
        new_fact = bundle["new_fact"]
        sentences = [f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n"]
        text_rephrase = bundle["generality"]["text_rephrase"][0]
        text_rephrase_text = f"{text_rephrase['prompt']} {text_rephrase['target']}".strip()
        sentences.append(f"New Fact: {new_fact}\nPrompt: {text_rephrase_text}\n\n")
        return sentences

    def _prepare_text_only_diverse_embeddings(self):
        if self.stored_example_bundles is None:
            raise RuntimeError("Text-only VICE baseline requires stored_example_bundles in the diverse cache.")
        sentences = []
        for bundle in self.stored_example_bundles:
            sentences.extend(self._bundle_to_text_only_sentences(bundle))
        embeddings = self.sentence_model.encode(sentences, convert_to_tensor=True)
        embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1).unsqueeze(-1)
        self.stored_text_only_sentences = sentences
        self.stored_text_only_embeddings = embeddings.cpu()
        print(f"Prepared text-only diverse embeddings: {len(sentences)} sentences from {len(self.stored_example_bundles)} bundles")

    def load_diverse_embeddings(self, cache_path: str):
        """Load precomputed diverse sentences + embeddings from .pt file."""
        data = torch.load(cache_path, map_location='cpu', weights_only=False)
        if 'stored_example_bundles' not in data and self.use_image_summary:
            raise RuntimeError(
                f"{cache_path} is missing stored_example_bundles. "
                f"Re-run preprocess_image_summaries.py to rebuild the diverse cache."
            )
        self.stored_sentences = data['stored_sentences']
        self.stored_embeddings = data['stored_embeddings']
        self.stored_example_bundles = data.get('stored_example_bundles')
        if self.image_summary_ablation != "none":
            self.stored_sentences = [
                self._apply_image_summary_ablation_to_sentence(sentence)
                for sentence in self.stored_sentences
            ]
            if self.stored_example_bundles is not None:
                self.stored_example_bundles = self._apply_image_summary_ablation_to_obj(self.stored_example_bundles)
            self.stored_embeddings = self.sentence_model.encode(self.stored_sentences, convert_to_tensor=True)
            self.stored_embeddings = self.stored_embeddings / torch.norm(self.stored_embeddings, p=2, dim=-1).unsqueeze(-1)
            self.stored_embeddings = self.stored_embeddings.cpu()
            print(f"Applied {self.image_summary_ablation} to diverse cache and rebuilt {len(self.stored_sentences)} embeddings")
        print(f"Loaded diverse embeddings from {cache_path}: {len(self.stored_sentences)} anchors")
        if not self.use_image_summary:
            if self.stored_example_bundles is not None:
                self._prepare_text_only_diverse_embeddings()
            else:
                self.stored_text_only_sentences = self.stored_sentences
                self.stored_text_only_embeddings = self.stored_embeddings.cpu()
                print(f"Using text-only diverse embeddings directly from {cache_path}")

    def load_image_summary_cache(self, *cache_paths):
        """Load precomputed image summaries from JSON files."""
        for path in cache_paths:
            if not path or not os.path.exists(path):
                continue
            with open(path, 'r') as f:
                entries = json.load(f)
            for e in entries:
                for image_path in self._image_path_cache_forms(e['image_path']):
                    self._image_summary_cache[(image_path, e['text'])] = e['summary']
        print(f"Loaded {len(self._image_summary_cache)} cached image summaries")

    @staticmethod
    def _image_path_cache_forms(image_path: str) -> List[str]:
        """Return stable cache keys for paths from this or a previous checkout."""
        if not image_path:
            return []

        normalized = os.path.normpath(os.fspath(image_path))
        forms = [normalized]
        if not os.path.isabs(normalized):
            forms.append(os.path.abspath(normalized))

        data_marker = f"{os.sep}data{os.sep}"
        marker_index = normalized.rfind(data_marker)
        if marker_index >= 0:
            forms.append(normalized[marker_index + 1:])

        return list(dict.fromkeys(forms))

    def generate_image_summary(self, text: str, image, image_path: str) -> str:
        # Try cache first (keyed by image file path + text)
        # Try multiple path forms: raw, absolute, and PIL filename
        candidate_paths = [
            image_path,
            os.path.abspath(image_path) if image_path else None,
            image.filename if hasattr(image, 'filename') else None,
        ]
        for candidate_path in candidate_paths:
            for path_form in self._image_path_cache_forms(candidate_path):
                if (path_form, text) in self._image_summary_cache:
                    return self._apply_image_summary_ablation(self._image_summary_cache[(path_form, text)])

        raise RuntimeError(
            "Image summary cache miss. Online summary generation is disabled.\n"
            "Re-run preprocess_image_summaries.py to cover this pair.\n"
            f"  image_path={image_path}\n"
            f"  text={text[:80]}..."
        )

    def _make_text_only_locality_sentence(self, bundle: Dict) -> str:
        text_loc = bundle["locality"]["text_loc"][0]
        prompt_text = f"{text_loc['prompt']} {text_loc['target']}".strip()
        return (
            "Image Summary: No image summary. This is a text-only locality example that should be preserved.\n"
            f"New Fact: {bundle['new_fact']}\n"
            f"Prompt: {prompt_text}\n\n"
        )

    def _get_role_sentence_from_bundle(self, bundle: Dict, role_name: str) -> str:
        if role_name == "reliability":
            return bundle["reliability"]["sentence"]
        if role_name == "text_rephrase":
            return bundle["generality"]["text_rephrase"][0]["sentence"]
        if role_name == "image_rephrase":
            return bundle["generality"]["image_rephrase"][0]["sentence"]
        if role_name == "locality_text":
            return bundle["locality"]["text_loc"][0].get("sentence", self._make_text_only_locality_sentence(bundle))
        if role_name == "locality_image":
            return bundle["locality"]["image_loc"][0]["sentence"]
        raise ValueError(f"Unsupported role_name={role_name}")

    def _build_icl_examples(self, request: Dict, prompt: str, target: str, top_k: int = 2) -> List[str]:
        """
        Retrieve in-context learning examples from diverse dataset (matching ike_main.py line 250).

        If diverse dataset is available, retrieves top-k similar sentences from it.
        Otherwise, constructs default examples.
        """
        if not self.use_image_summary:
            new_fact = f"{prompt} {target}".strip()
            query_sentence = f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n"
            if self.stored_text_only_embeddings is None or self.stored_text_only_sentences is None:
                raise RuntimeError("Text-only diverse embeddings are not prepared.")
            query_embedding = self.sentence_model.encode(query_sentence, convert_to_tensor=True).unsqueeze(0).to(self.device[0])
            query_embedding = query_embedding / torch.norm(query_embedding, p=2, dim=-1)
            stored_emb = self.stored_text_only_embeddings.to(self.device[0])
            hits = util.semantic_search(
                query_embedding,
                stored_emb,
                score_function=util.dot_score,
                top_k=min(2, len(self.stored_text_only_sentences))
            )
            return [self.stored_text_only_sentences[hit["corpus_id"]] for hit in hits[0]]

        new_fact = f"{prompt} {target}".strip()
        image = request['image']
        image_path = request['image_path']
        image_summary = self.generate_image_summary(new_fact, image, image_path)
        query_sentence = f"Image Summary: {image_summary}\nNew Fact: {new_fact}\nPrompt: {new_fact}\n\n"

        # If diverse embeddings are available, retrieve from them
        if self.stored_embeddings is not None and len(self.stored_sentences) > 0:
            query_embedding = self.sentence_model.encode(query_sentence, convert_to_tensor=True).unsqueeze(0).to(self.device[0])
            query_embedding = query_embedding / torch.norm(query_embedding, p=2, dim=-1)

            stored_emb = self.stored_embeddings.to(self.device[0])
            hits = util.semantic_search(
                query_embedding,
                stored_emb,
                score_function=util.dot_score,
                top_k=min(top_k, len(self.stored_sentences))
            )

            icl_examples = []
            for hit, role_name in zip(hits[0], self.icl_role_order):
                bundle = self.stored_example_bundles[hit["corpus_id"]]
                context_sentence = self._get_role_sentence_from_bundle(bundle, role_name)
                icl_examples.append(context_sentence)
            return icl_examples
        else:
            raise RuntimeError("No diverse dataset embeddings available for ICL examples retrieval")

    def edit_one_piece(self, request: Dict):
        """
        Add a new edit to sequence memory (matching original VICE.py).

        request = {'prompt': str, 'target_new': str, 'generality': {...}, ...}
        Retrieves icl_examples from diverse dataset using semantic search.
        """
        prompt = request['prompt']
        target = request['target_new']

        if prompt and target:
            if self.use_image_summary:
                icl_examples = self._build_icl_examples(request, prompt, target, top_k=len(self.icl_role_order))
                self.add_to_sequence_memory(prompt, target, icl_examples, image=request.get('image'), image_path=request.get('image_path'))
            else:
                icl_examples = self._build_icl_examples(request, prompt, target, top_k=2)
                self.add_to_sequence_memory(prompt, target, icl_examples)
        else:
            raise ValueError("Both prompt and target must be non-empty for editing")

    def edit_batch(self, requests: List[Dict]):
        """Add multiple edits to sequence memory."""
        for request in requests:
            self.edit_one_piece(request)

    def add_to_sequence_memory(self, prompt: str, target: str, icl_examples: Optional[List[str]] = None, image=None, image_path: str = None):
        """
        Add new fact to sequence memory with embedding.
        """
        new_fact = f"{prompt} {target}".strip()
        if self.use_image_summary:
            image_summary = self.generate_image_summary(prompt, image, image_path)
            retrieval_sentence = f"Image Summary: {image_summary}\nNew Fact: {prompt}\nPrompt: {prompt}\n\n"
            fact_sentence = f"Image Summary: {image_summary}\nNew Fact: {new_fact}\nPrompt: {new_fact}\n\n"
        else:
            retrieval_sentence = f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n"
            fact_sentence = f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n"
        query_embedding = self.sentence_model.encode(retrieval_sentence, convert_to_tensor=True).unsqueeze(0).cpu()
        context = "".join(icl_examples) if icl_examples is not None else ""

        self.sequence_memory["new_facts"].append(new_fact)
        self.sequence_memory["contexts"].append(context)
        self.sequence_memory["fact_sentences"].append(fact_sentence)
        self.sequence_memory["embeddings"].append(query_embedding)
        if self.use_image_summary:
            img_embedding = self._encode_image(image, image_path).cpu()
            self.sequence_memory["img_embeddings"].append(img_embedding)
        scope_embedding = self._encode_scope(prompt, image, image_path)
        if scope_embedding is not None:
            self.sequence_memory["scope_embeddings"].append(scope_embedding.cpu())

        print(f"Added to sequence memory: {new_fact[:50]}...")
        return self.sequence_memory

    def retrieve_from_sequence_memory(self, prompt: str, target: str = "", top_k: int = 1, image=None, image_path: str = None) -> List[Dict]:
        """
        Retrieve memory using scope classifier when available, otherwise fallback to sentence embeddings.
        """
        if len(self.sequence_memory["new_facts"]) == 0:
            return []

        use_scope = (self.scope_projection_head is not None
                     and len(self.sequence_memory.get("scope_embeddings", [])) == len(self.sequence_memory["new_facts"]))

        if use_scope:
            query_emb = self._encode_scope(prompt, image, image_path)
            if query_emb is None:
                return []
            memory_embs = torch.cat(self.sequence_memory["scope_embeddings"], dim=0).to(self.device[0])
            memory_embs = memory_embs / memory_embs.norm(p=2, dim=-1).unsqueeze(-1)
            scores = memory_embs @ query_emb.squeeze(0)
            k = min(top_k, scores.shape[0])
            top_scores, top_indices = torch.topk(scores, k=k)
            return [{
                "new_fact": self.sequence_memory["new_facts"][int(i)],
                "context": self.sequence_memory["contexts"][int(i)],
                "fact_sentence": self.sequence_memory["fact_sentences"][int(i)],
                "t_score": None,
                "i_score": float(s),
                "score": float(s),
            } for s, i in zip(top_scores, top_indices)]

        # Fallback: sentence-transformer retrieval
        if self.use_image_summary and image is not None:
            image_summary = self.generate_image_summary(prompt, image, image_path)
            query_sentence = f"Image Summary: {image_summary}\nNew Fact: {prompt}\nPrompt: {prompt}\n\n"
        else:
            query_sentence = f"New Fact: {prompt}\nPrompt: {prompt}\n\n"
        query_embedding = self.sentence_model.encode(query_sentence, convert_to_tensor=True).unsqueeze(0).to(self.device[0])
        query_embedding = query_embedding / torch.norm(query_embedding, p=2, dim=-1)

        memory_embeddings = torch.cat(self.sequence_memory["embeddings"], dim=0).to(self.device[0])
        memory_embeddings = memory_embeddings / torch.norm(memory_embeddings, p=2, dim=-1).unsqueeze(-1)
        hits = util.semantic_search(
            query_embedding, memory_embeddings, score_function=util.dot_score,
            top_k=min(top_k, memory_embeddings.shape[0]),
        )
        return [{
            "new_fact": self.sequence_memory["new_facts"][h["corpus_id"]],
            "context": self.sequence_memory["contexts"][h["corpus_id"]],
            "fact_sentence": self.sequence_memory["fact_sentences"][h["corpus_id"]],
            "t_score": float(h["score"]),
            "i_score": None,
            "score": float(h["score"]),
        } for h in hits[0]]

    def build_inference_context(self, prompt: str, target: str = "", top_k: int = 1, image=None, image_path: str = None) -> Tuple[str, List[float]]:
        """Build context string from retrieved memory items."""
        if getattr(self, 'no_icl', False):
            return "", [], [], []
        retrieved = self.retrieve_from_sequence_memory(prompt, target, top_k=top_k, image=image, image_path=image_path)
        if len(retrieved) == 0:
            return "", [], [], []
        item = retrieved[0]
        context = (
            f"{item['context']}"
            f"{item['fact_sentence']}"
            "Image Summary: Use the given real image as the visual evidence for this query.\n"
            f"New Fact: {item['new_fact']}\n"
            f"Prompt: {prompt}"
        )
        return context, [item['score'] for item in retrieved], [item['t_score'] for item in retrieved], [item['i_score'] for item in retrieved]

    def _augmented_prompts_imgs_target_to_xym(self, prompts: List[str], images, targets: List[str],
                                               return_extra_info: bool = False, image_paths: List[str] = None, **kwargs):
        """
        Wrapper for vllm.prompts_imgs_target_to_xym() that automatically augments prompts
        with retrieved sequence memory context before inference.

        Gating: binary classifier has the final say. Falls back to sim_threshold if no binary head.
        """
        augmented_prompts = []
        extra_info = {"original_prompts": prompts, "targets": targets}
        gate_decisions, all_scores = [], []
        img_iter = images if images is not None else [None] * len(prompts)
        path_iter = image_paths if image_paths is not None else [None] * len(prompts)
        for prompt, target, image, image_path in zip(prompts, targets, img_iter, path_iter):
            prefix, scores, t_scores, i_scores = self.build_inference_context(prompt, target, top_k=1, image=image, image_path=image_path)

            if not scores:
                use_prefix, gate_reason = False, "no_retrieval"
            elif self.scope_projection_head is not None:
                i_ok = i_scores[0] >= self.sim_threshold
                use_prefix, gate_reason = (True, "scope_sim_ok") if i_ok else (False, "scope_sim_low")
            else:
                t_ok = max(t_scores) >= self.sim_threshold
                use_prefix, gate_reason = (True, "sent_sim_ok") if t_ok else (False, "sent_sim_low")

            augmented_prompts.append(prefix if (use_prefix and prefix) else prompt)
            gate_decisions.append(gate_reason)
            all_scores.append((scores, t_scores, i_scores))

        base_result = self._orig_prompts_imgs_target_to_xym(
            augmented_prompts, images, targets, return_extra_info=return_extra_info, **kwargs)
        if return_extra_info:
            extra_info["augmented_prompts"] = augmented_prompts
            extra_info["scores"] = all_scores
            extra_info["gate_decisions"] = gate_decisions
            return base_result[:3] + ({"vice": extra_info},)
        return base_result

    def materialize_pending_edits(self):
        """No pending edits in VICE - all edits are immediately materialized to memory."""
        pass

    def save_ckpt_eval(self, eval_cfg, ckpt_path: str):
        ckpt = {"eval_cfg": eval_cfg, "sequence_memory": self.sequence_memory}
        if self.scope_projection_head is not None:
            ckpt["scope_classifier"] = {
                "projection_state_dict": self.scope_projection_head.state_dict(),
                "feature_dim": self.scope_classifier_feature_dim,
                "projection_dim": self.scope_classifier_projection_dim,
                "temperature": self.scope_classifier_temperature,
            }
        torch.save(ckpt, ckpt_path)

    def load_ckpt(self, ckpt_path: str, device=None, restrict=True):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.sequence_memory = ckpt["sequence_memory"]
        self.sequence_memory.setdefault("scope_embeddings", [])
        self.sequence_memory.setdefault("img_embeddings", [])
        self._scope_embedding_cache = {}
        scope_ckpt = ckpt.get("scope_classifier")
        if scope_ckpt is not None:
            projection = torch.nn.Sequential(
                torch.nn.Linear(scope_ckpt["feature_dim"], scope_ckpt["feature_dim"]),
                torch.nn.ReLU(),
                torch.nn.Linear(scope_ckpt["feature_dim"], scope_ckpt["projection_dim"]),
            ).to(self.device[0])
            projection.load_state_dict(scope_ckpt["projection_state_dict"], strict=restrict)
            projection.eval()
            self.scope_projection_head = projection
            self.scope_classifier_temperature = scope_ckpt.get("temperature", 0.07)
            self.scope_classifier_feature_dim = scope_ckpt["feature_dim"]
            self.scope_classifier_projection_dim = scope_ckpt["projection_dim"]
            self._scope_embedding_cache = {}
