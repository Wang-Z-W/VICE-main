"""
Preprocess cached image summaries and diverse embeddings for VICE-VL.
"""

import json
import os

import torch
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from utils.GLOBAL import ROOT_PATH
from dataset.vllm import EVQA, EIC, VLKEB, WaterBird, WaterBirdV2

# ── Config ──────────────────────────────────────────────────────────────────
NUM_GPUS = 8
SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"

_DATASET_CFG = {
    "EVQA": {
        "cls": EVQA,
        "train_json": "data/easy-edit-mm/vqa/vqa_train.json",
        "eval_json":  "data/easy-edit-mm/vqa/vqa_eval.json",
        "img_dir":    "data/easy-edit-mm/images",
        "cache_dir":  "data/easy-edit-mm/vqa",
    },
    "EIC": {
        "cls": EIC,
        "train_json": "data/easy-edit-mm/caption/caption_train_edit.json",
        "eval_json":  "data/easy-edit-mm/caption/caption_eval_edit.json",
        "img_dir":    "data/easy-edit-mm/images",
        "cache_dir":  "data/easy-edit-mm/caption",
    },
    "VLKEB": {
        "cls": VLKEB,
        "train_json": "data/VLKEB/train.json",
        "eval_json":  "data/VLKEB/eval.json",
        "img_dir":    "data/VLKEB/mmkb_images",
        "cache_dir":  "data/VLKEB",
    },
    "WATERBIRD": {
        "cls": WaterBird,
        "train_json": "data/WaterBird/edit_annotations_truelabel_balanced.json",
        "eval_json":  "data/WaterBird/edit_annotations_truelabel_balanced.json",
        "img_dir":    "data/WaterBird",
        "cache_dir":  "data/WaterBird",
        "split":      "train",
        "eval_split": "test",
    },
    "WATERBIRDV2": {
        "cls": WaterBirdV2,
        "train_json": "data/WaterBird-v2.0/edit_annotations.json",
        "eval_json":  "data/WaterBird-v2.0/edit_annotations.json",
        "img_dir":    "",
        "cache_dir":  "data/WaterBird-v2.0",
        "split":      "train",
        "eval_split": "test",
    },
}


# ── LLaVA model (same backbone as editing) ──────────────────────────────────
def load_llava(device):
    from transformers import LlavaForConditionalGeneration, LlavaProcessor
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        revision="a272c74",
        device_map=device,
        # max_memory = max_memory
    )
    processor = LlavaProcessor.from_pretrained(model_id, revision="a272c74")
    model = model.eval().requires_grad_(False)
    return model, processor


def _normalize_image_path(image):
    return image.filename if hasattr(image, 'filename') else image


def _fact_text(prompt, target):
    return f"{prompt} {target}".strip()


def _format_summary_block(summary, new_fact, prompt_text):
    return f"Image Summary: {summary}\nNew Fact: {new_fact}\nPrompt: {prompt_text}\n\n"


def generate_image_summary(model, processor, text: str, image, device: str) -> str:
    """Text-guided visual summary using LLaVA (same backbone as editing)."""
    prompt = (
        "USER: <image>\n"
        "Given the image and the text statement, write a concise visual summary for knowledge editing. "
        "The summary must identify the target-related visual evidence, describe other salient objects or attributes "
        "that could act as distractors, and briefly contrast the target evidence against those distractors. "
        "Do not answer the question directly. Do not simply copy the text statement.\n"
        "Return exactly three lines in the following format:\n"
        "Target Evidence: ...\n"
        "Distractors: ...\n"
        "Contrast: ...\n"
        f"Text Statement: {text}\n"
        "ASSISTANT:"
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    gen_ids = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.decode(gen_ids[0], skip_special_tokens=True).strip()


# ── Collect (image_path, text) pairs ────────────────────────────────────────
def collect_pairs(data_path, img_root_dir, data_n, dataset_cls=EVQA, split=None):
    if split is not None:
        dataset = dataset_cls(data_path, img_root_dir, split, data_n)
    else:
        dataset = dataset_cls(data_path, img_root_dir, data_n)
    pairs = set()
    for d in dataset.data_with_img_path:
        img_path = d['requests'][0]['image']
        prompt = d['requests'][0]['prompt']
        target = d['requests'][0]['target_new']
        text = _fact_text(prompt, target)
        if img_path is not None:
            pairs.add((img_path, text))
        for g in d['generality']['text_rephrase']:
            if g['image'] is not None:
                pairs.add((g['image'], _fact_text(g['prompt'], g['target'])))
        for g in d['generality']['image_rephrase']:
            if g['image'] is not None:
                pairs.add((g['image'], _fact_text(g['prompt'], g['target'])))
        for l in d['locality']['image_loc']:
            if l['image'] is not None:
                pairs.add((l['image'], _fact_text(l['prompt'], l['target'])))
    return list(pairs)


def collect_edit_pairs(data_path, img_root_dir, data_n, dataset_cls=EVQA, split=None):
    if split is not None:
        dataset = dataset_cls(data_path, img_root_dir, split, data_n)
    else:
        dataset = dataset_cls(data_path, img_root_dir, data_n)
    pairs = set()
    for d in dataset.data_with_img_path:
        request = d['requests'][0]
        if request['image'] is not None:
            pairs.add((request['image'], _fact_text(request['prompt'], request['target_new'])))
    return list(pairs)


def collect_query_pairs(data_path, img_root_dir, data_n, dataset_cls=EVQA, split=None):
    if split is not None:
        dataset = dataset_cls(data_path, img_root_dir, split, data_n)
    else:
        dataset = dataset_cls(data_path, img_root_dir, data_n)
    pairs = set()
    for d in dataset.data_with_img_path:
        request = d['requests'][0]
        if request['image'] is not None:
            pairs.add((request['image'], request['prompt']))
        for g in d['generality']['text_rephrase']:
            if g['image'] is not None:
                pairs.add((g['image'], g['prompt']))
        for g in d['generality']['image_rephrase']:
            if g['image'] is not None:
                pairs.add((g['image'], g['prompt']))
        for l in d['locality']['image_loc']:
            if l['image'] is not None:
                pairs.add((l['image'], l['prompt']))
    return list(pairs)


# ── Worker process for multi-GPU generation ─────────────────────────────────
def _worker(rank, pairs_chunk, return_dict):
    """Each worker loads LLaVA on its own GPU and processes a chunk of pairs."""
    device = f"cuda:{rank}"
    model, processor = load_llava(device)
    results = []
    desc = f"GPU {rank}"
    for img_path, text in tqdm(pairs_chunk, desc=desc, position=rank):
        image = Image.open(img_path).convert("RGB")
        summary = generate_image_summary(model, processor, text, image, device)
        results.append({"image_path": img_path, "text": text, "summary": summary})
        image.close()
    return_dict[rank] = results


def build_summary_cache_parallel(pairs, output_path, num_gpus):
    """Generate image summaries in parallel across multiple GPUs."""
    existing_cache = []
    existing_cache_map = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing_cache = json.load(f)
        existing_cache_map = {(e['image_path'], e['text']): e['summary'] for e in existing_cache}
        missing_pairs = [pair for pair in pairs if pair not in existing_cache_map]
        if len(missing_pairs) == 0:
            print(f"Skip existing summary cache: {output_path}")
            return existing_cache_map
        print(f"Patch existing summary cache: {output_path} | missing {len(missing_pairs)} / {len(pairs)}")
        pairs = missing_pairs

    # Split pairs into chunks, one per GPU
    chunks = [[] for _ in range(num_gpus)]
    for i, pair in enumerate(pairs):
        chunks[i % num_gpus].append(pair)

    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for rank in range(num_gpus):
        if len(chunks[rank]) == 0:
            continue
        p = mp.Process(target=_worker, args=(rank, chunks[rank], return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Summary worker failed: pid={p.pid}, exitcode={p.exitcode}")

    # Merge results in rank order to keep deterministic ordering
    cache = list(existing_cache)
    for rank in range(num_gpus):
        if rank in return_dict:
            cache.extend(return_dict[rank])

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(cache)} summaries to {output_path}")
    return {(e['image_path'], e['text']): e['summary'] for e in cache}


# ── Diverse embeddings ──────────────────────────────────────────────────────
def build_diverse_embeddings(train_data_path, img_root_dir, summary_cache,
                             sentence_model, device, output_path, use_kmeans,
                             dataset_cls=EVQA, split=None):
    if os.path.exists(output_path):
        print(f"Skip existing diverse embeddings: {output_path}")
        return

    from transformers import CLIPModel, CLIPProcessor

    if split is not None:
        dataset = dataset_cls(train_data_path, img_root_dir, split, None)
    else:
        dataset = dataset_cls(train_data_path, img_root_dir, None)
    data_list = dataset.data_with_img

    if use_kmeans:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_model.eval()
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        img_features_list, txt_features_list = [], []
        with torch.no_grad():
            for data in tqdm(data_list, desc="Computing CLIP features for KMeans"):
                prompt = data["requests"][0]["prompt"]
                text_inputs = clip_processor(text=prompt, return_tensors="pt",
                                             truncation=True, max_length=77, padding=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                txt_features_list.append(clip_model.get_text_features(**text_inputs).squeeze(0).cpu())

                img = data["requests"][0]["image"]
                image_inputs = clip_processor(images=img, return_tensors="pt")
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                img_features_list.append(clip_model.get_image_features(**image_inputs).squeeze(0).cpu())

        img_features = torch.stack(img_features_list)
        txt_features = torch.stack(txt_features_list)
        img_features = img_features / torch.norm(img_features, p=2, dim=-1, keepdim=True)
        txt_features = txt_features / torch.norm(txt_features, p=2, dim=-1, keepdim=True)
        features = torch.cat((img_features, txt_features), dim=-1)
        features = features / torch.norm(features, p=2, dim=-1, keepdim=True)

        num_clusters = max(1, len(data_list) // 20)
        print(f"KMeans: {len(data_list)} samples -> {num_clusters} clusters")
        km = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40,
                    init='k-means++', random_state=1993)
        cluster_ids = km.fit_predict(features.numpy())

        selected_indices = []
        np.random.seed(1993)
        for cid in range(num_clusters):
            indices = np.where(cluster_ids == cid)[0]
            if len(indices) > 0:
                selected_indices.append(indices[np.random.permutation(len(indices))[0]])
        print(f"Selected {len(selected_indices)} diverse samples")
        data_list = [data_list[i] for i in selected_indices]

        del clip_model, clip_processor
        torch.cuda.empty_cache()

    sentences = []
    bundles = []
    for data in tqdm(data_list, desc="Building diverse sentences"):
        img_path = _normalize_image_path(data['requests'][0]['image'])
        prompt = data['requests'][0]['prompt']
        target = data['requests'][0]['target_new']
        new_fact = _fact_text(prompt, target)

        summary = summary_cache.get((img_path, new_fact))
        if summary is None:
            raise RuntimeError(f"Missing summary cache for ({img_path}, {new_fact[:60]}...)")
        anchor_sentence = _format_summary_block(summary, new_fact, new_fact)
        sentences.append(anchor_sentence)
        bundle = {
            "new_fact": new_fact,
            "reliability": {
                "image": img_path,
                "prompt": prompt,
                "target": target,
                "sentence": anchor_sentence,
            },
            "generality": {"text_rephrase": [], "image_rephrase": []},
            "locality": {"text_loc": [], "image_loc": []},
        }

        text_rephrase = data['generality']['text_rephrase'][0]
        text_rephrase_text = _fact_text(text_rephrase['prompt'], text_rephrase['target'])
        text_rephrase_summary = summary_cache.get((img_path, text_rephrase_text))
        if text_rephrase_summary is None:
            raise RuntimeError(
                f"Missing summary cache for ({img_path}, {text_rephrase_text[:60]}...)"
            )
        bundle["generality"]["text_rephrase"].append(
            {
                "image": img_path,
                "prompt": text_rephrase['prompt'],
                "target": text_rephrase['target'],
                "sentence": _format_summary_block(text_rephrase_summary, new_fact, text_rephrase_text),
            }
        )

        image_rephrase = data['generality']['image_rephrase'][0]
        image_rephrase_path = _normalize_image_path(image_rephrase['image'])
        image_rephrase_text = _fact_text(image_rephrase['prompt'], image_rephrase['target'])
        image_rephrase_summary = summary_cache.get((image_rephrase_path, image_rephrase_text))
        if image_rephrase_summary is None:
            raise RuntimeError(
                f"Missing summary cache for ({image_rephrase_path}, {image_rephrase_text[:60]}...)"
            )
        bundle["generality"]["image_rephrase"].append(
            {
                "image": image_rephrase_path,
                "prompt": image_rephrase['prompt'],
                "target": image_rephrase['target'],
                "sentence": _format_summary_block(image_rephrase_summary, new_fact, image_rephrase_text),
            }
        )

        text_loc = data['locality']['text_loc'][0]
        text_loc_text = _fact_text(text_loc['prompt'], text_loc['target'])
        bundle["locality"]["text_loc"].append(
            {
                "image": None,
                "prompt": text_loc['prompt'],
                "target": text_loc['target'],
                "sentence": _format_summary_block(
                    "No image summary. This is a text-only locality example that should be preserved.",
                    new_fact,
                    text_loc_text,
                ),
            }
        )

        image_loc = data['locality']['image_loc'][0]
        image_loc_path = _normalize_image_path(image_loc['image'])
        image_loc_text = _fact_text(image_loc['prompt'], image_loc['target'])
        image_loc_summary = summary_cache.get((image_loc_path, image_loc_text))
        if image_loc_summary is None:
            raise RuntimeError(
                f"Missing summary cache for ({image_loc_path}, {image_loc_text[:60]}...)"
            )
        bundle["locality"]["image_loc"].append(
            {
                "image": image_loc_path,
                "prompt": image_loc['prompt'],
                "target": image_loc['target'],
                "sentence": _format_summary_block(image_loc_summary, new_fact, image_loc_text),
            }
        )
        bundles.append(bundle)

    embeddings = sentence_model.encode(sentences, convert_to_tensor=True)
    embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(
        {
            'stored_sentences': sentences,
            'stored_embeddings': embeddings,
            'stored_example_bundles': bundles,
        },
        output_path,
    )
    print(f"Saved diverse embeddings: {len(sentences)} anchors -> {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess image summaries for VICE-VL")
    parser.add_argument("--data_name", type=str, default="all",
                        choices=["evqa", "eic", "vlkeb", "waterbird", "waterv2", "all"],
                        help="Dataset name (default: all)")
    parser.add_argument("--skip_summaries", action="store_true",
                        help="Skip image summary generation (steps 1-3)")
    parser.add_argument("--skip_diverse", action="store_true",
                        help="Skip diverse embedding generation (step 4)")
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    data_names = ["EVQA", "EIC", "VLKEB"] if args.data_name == "all" else [args.data_name.upper()]

    # Map waterv2 → WATERBIRDV2
    data_names = ["WATERBIRDV2" if n == "WATERV2" else n for n in data_names]

    for data_name in data_names:
        cfg = _DATASET_CFG[data_name]
        dataset_cls = cfg["cls"]
        train_json = os.path.join(ROOT_PATH, cfg["train_json"])
        eval_json = os.path.join(ROOT_PATH, cfg["eval_json"])
        img_dir = os.path.join(ROOT_PATH, cfg["img_dir"])
        cache_dir = os.path.join(ROOT_PATH, cfg["cache_dir"])
        split = cfg.get("split", None)

        print(f"\n{'='*60}")
        print(f"=== Dataset: {data_name} ===")
        print(f"{'='*60}")

        if not args.skip_summaries:
            print("=== Step 1: Train image summaries ===")
            train_pairs = collect_pairs(train_json, img_dir, None, dataset_cls, split=split)
            print(f"Collected {len(train_pairs)} unique (image, text) pairs")
            train_cache = build_summary_cache_parallel(
                train_pairs,
                os.path.join(cache_dir, 'image_summaries_cache_train.json'),
                NUM_GPUS,
            )

            eval_split = cfg.get("eval_split", split)
            print("=== Step 2: Edit image summaries patch ===")
            eval_pairs = collect_edit_pairs(eval_json, img_dir, None, dataset_cls, split=eval_split)
            print(f"Collected {len(eval_pairs)} edit (image, text) pairs")
            build_summary_cache_parallel(
                eval_pairs,
                os.path.join(cache_dir, 'image_summaries_cache_eval_edit.json'),
                NUM_GPUS,
            )

            print("=== Step 3: Eval query image summaries patch ===")
            eval_query_pairs = collect_query_pairs(eval_json, img_dir, None, dataset_cls, split=eval_split)
            print(f"Collected {len(eval_query_pairs)} eval query (image, prompt) pairs")
            build_summary_cache_parallel(
                eval_query_pairs,
                os.path.join(cache_dir, 'image_summaries_cache_eval_query.json'),
                NUM_GPUS,
            )
        else:
            train_cache_map = None
            train_cache_path = os.path.join(cache_dir, 'image_summaries_cache_train.json')
            if os.path.exists(train_cache_path):
                with open(train_cache_path) as f:
                    train_cache_map = {(e['image_path'], e['text']): e['summary'] for e in json.load(f)}
                print(f"Loaded existing train cache: {train_cache_path} ({len(train_cache_map)} entries)")
            train_cache = train_cache_map

        if not args.skip_diverse:
            print("=== Step 4: Diverse embeddings ===")
            if train_cache is None:
                train_cache_path = os.path.join(cache_dir, 'image_summaries_cache_train.json')
                if not os.path.exists(train_cache_path):
                    raise RuntimeError(f"Train cache not found: {train_cache_path}. Run without --skip_summaries first.")
                with open(train_cache_path) as f:
                    train_cache = {(e['image_path'], e['text']): e['summary'] for e in json.load(f)}

            # Normalise any relative image paths to absolute so they match
            # PIL's .filename (which resolves to an absolute path on open).
            _abs_cache = {}
            for (img_path, text), summary in list(train_cache.items()):
                if not os.path.isabs(img_path):
                    img_path = os.path.normpath(os.path.join(ROOT_PATH, img_path))
                _abs_cache[(img_path, text)] = summary
            train_cache = _abs_cache

            diverse_prefix = os.path.basename(cache_dir)  # e.g. "vqa", "caption", "VLKEB"
            sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME, device="cuda:0")
            build_diverse_embeddings(
                train_json, img_dir, train_cache,
                sentence_model, "cuda:0",
                os.path.join(cache_dir, f'{diverse_prefix}_train_diverse.pt'),
                False, dataset_cls, split=split,
            )
            build_diverse_embeddings(
                train_json, img_dir, train_cache,
                sentence_model, "cuda:0",
                os.path.join(cache_dir, f'{diverse_prefix}_train_diverse_kmeans.pt'),
                True, dataset_cls, split=split,
            )

    print("\n=== All Done ===")
