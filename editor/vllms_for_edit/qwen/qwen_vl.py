from typing import List, Optional
from ..base import BaseVLLMForEdit
from PIL.Image import Image as ImageClass
import torch


class Qwen2VLForEdit(BaseVLLMForEdit):
    '''For Qwen2-VL-7B-Instruct'''
    def __init__(self, model_path: str, device: List[str] = ['cuda'],
                 auto_add_img_special_token=True) -> None:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        # Qwen2VL uses device_map="auto" by default; for single-device editing,
        # load directly on the target device to avoid multi-GPU distribution.
        # We load in bfloat16 for memory efficiency.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device[0] if len(device) == 1 else None,
            trust_remote_code=True,
        )
        if len(device) > 1:
            # For multi-GPU, let the model decide device_map
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.image_processor.max_pixels = 1003520  # ~1000x1000, avoid OOM on large images
        self.processor.image_processor.min_pixels = 1003520
        self.model = self.model.eval().requires_grad_(False)
        self.device_str = device[0]
        super().__init__(self.model, device, auto_add_img_special_token)

    def get_llm_tokenizer(self):
        return self.processor.tokenizer

    def get_llm_input_embeds(self, texts: List[str], imgs: Optional[List[ImageClass]] = None):
        if imgs is not None:
            # Qwen2VL processor expects images and text
            inputs = self.processor(
                text=texts,
                images=imgs,
                padding=True,
                return_tensors='pt',
            )
            for k, v in inputs.items():
                if hasattr(v, 'to'):
                    inputs[k] = v.to(self.device_str)

            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            pixel_values = inputs.get('pixel_values', None)
            image_grid_thw = inputs.get('image_grid_thw', None)

            # 1. Get token embeddings from the language model
            inputs_embeds = self.model.model.embed_tokens(input_ids)

            # 2. Get visual features if images present
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.model.visual.get_dtype())
                image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

                # 3. Merge: replace image token positions with vision features
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # 4. Build vt_range from actual image token positions
            if imgs is not None:
                img_positions = torch.where(input_ids[0] == self.config.image_token_id)[0]
                if len(img_positions) > 0:
                    img_begin = int(img_positions[0])
                    img_end = int(img_positions[-1]) + 1
                    vt_range = [img_begin, img_end]
                else:
                    vt_range = None
            else:
                vt_range = None

            llm_inpt = {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
            }
        else:
            # Text-only: no image processing needed
            inputs = self.processor(
                text=texts,
                padding=True,
                return_tensors='pt',
            )
            for k, v in inputs.items():
                if hasattr(v, 'to'):
                    inputs[k] = v.to(self.device_str)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            inputs_embeds = self.model.model.embed_tokens(input_ids)
            llm_inpt = {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
            }
            vt_range = None

        return llm_inpt, vt_range

    def get_llm_outpt(self, llm_inpt, vt_range=None):
        assert 'inputs_embeds' in llm_inpt.keys()
        # Forward through Qwen2Decoder (language model only, no vision)
        outputs = self.model.model(
            inputs_embeds=llm_inpt['inputs_embeds'],
            attention_mask=llm_inpt.get('attention_mask', None),
            use_cache=False,
        )
        # Apply LM head
        logits = self.model.lm_head(outputs[0])
        class Output:
            def __init__(self, logits):
                self.logits = logits
        return Output(logits)

    def get_img_special_token_str(self):
        return '<|image_pad|>'

    def get_img_special_token_id(self):
        return self.config.image_token_id

    def get_img_token_n(self):
        # Qwen2VL generates a variable number of image tokens per image depending on input resolution.
        # The exact count is spatial_patch_size**2 * temporal_patch_size * (merge_size**2) for a standard image.
        # For 224x224: 16*16 * 2 * 4 = 2048, but this varies with resolution.
        # We compute it dynamically from the actual image embeddings.
        return self.config.vision_config.spatial_patch_size ** 2 * \
               self.config.vision_config.temporal_patch_size * \
               self.config.vision_config.spatial_merge_size ** 2

    def is_q_former_based(self):
        return False

    @property
    def config(self):
        return self.model.config
