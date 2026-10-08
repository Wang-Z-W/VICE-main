from typing import List, Union, Optional
from ..base import BaseVLLMForEdit, get_multiple_gpus_for_vllm
from PIL.Image import Image as ImageClass
from transformers import  AutoTokenizer
import torch, inspect
import types
from functools import wraps
from transformers.modeling_outputs import CausalLMOutputWithPast
from torchvision.transforms import ToPILImage


class LlavaForEdit(BaseVLLMForEdit):
    '''For llava-v1.5-7b-hf'''
    def __init__(self, model_path:str, device:List[str] = ['cuda'], max_memory = None,
                 auto_add_img_special_token = True) -> None:
        from transformers import LlavaForConditionalGeneration, LlavaProcessor
        device_map = get_multiple_gpus_for_vllm(model_path, device)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            revision="a272c74",
            device_map = device_map,
            # max_memory = max_memory
        )
        self.processor = LlavaProcessor.from_pretrained(model_path, revision="a272c74")
        # self.processor.patch_size = self.model.config.vision_config.patch_size
        # self.processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        self.model = self.model.eval().requires_grad_(False)
        super().__init__(self.model, device, auto_add_img_special_token)

    def get_llm_tokenizer(self):
        return self.processor.tokenizer

    def _wrap_full_model_forward(self):
        model = self.model  # LlavaForConditionalGeneration
        if getattr(model, "_full_forward_wrapped", False):
            return
        model._forward_orig = model.forward

        @wraps(model._forward_orig)
        def wrapped_forward(module_self, *args, **kwargs):
            # 1. 调用原始 forward 前，先用 processor/generate 的逻辑把输入拆出来
            text, images = self._extract_text_and_imgs(args, kwargs)
            llm_inpt, vt_range = self.get_llm_input_embeds([text], [images])
            # 2. 用包装过的 get_llm_outpt 完成中间层 hook + MoE
            outputs = self.get_llm_outpt(llm_inpt, vt_range)
            # 3. 组织返回对象，保持与原 forward 一致
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = None
            if kwargs.get("labels") is not None:
                loss = module_self.loss(logits, kwargs["labels"])
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)

        model.forward = types.MethodType(wrapped_forward, model)
        model._full_forward_wrapped = True

    def _extract_text_and_imgs(self, args, kwargs):
        """
        从 `generate` 的输入中恢复 prompt 文本与图像，用于重新构造
        `get_llm_input_embeds` 所需的数据。
        """
        forward_fn = getattr(self.model, "_forward_orig", self.model.forward)
        sig = inspect.signature(forward_fn)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        input_ids = bound.arguments.get("input_ids")
        if input_ids is None:
            raise ValueError("LLaVA forward without `input_ids` is not supported in generate flow.")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        prompt = self.processor.tokenizer.batch_decode(
            input_ids.detach().cpu(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]

        pixel_values = bound.arguments.get("pixel_values")
        image = None
        if pixel_values is not None:
            if isinstance(pixel_values, torch.Tensor):
                px = pixel_values.detach().cpu()
            else:
                px = torch.as_tensor(pixel_values)
            if px.dim() == 3:
                px = px.unsqueeze(0)

            image_processor = self.processor.image_processor
            if hasattr(image_processor, "postprocess"):
                image = image_processor.postprocess(px, output_type="pil")[0]
            elif hasattr(image_processor, "post_process"):
                image = image_processor.post_process(px, output_type="pil")[0]
            else:
                mean = torch.tensor(image_processor.image_mean).view(1, -1, 1, 1)
                std = torch.tensor(image_processor.image_std).view(1, -1, 1, 1)
                unnorm = (px * std + mean).clamp(0, 1)
                image = ToPILImage()(unnorm[0])

        return prompt, image


    def get_llm_input_embeds(self, texts:List[str], imgs:Optional[List[ImageClass]] = None):
        '''Only support one image in one text.'''
        def get_llava_llm_inpt(mllm, input_ids, attention_mask, pixel_values):
            position_ids = None
            vision_feature_layer = mllm.config.vision_feature_layer
            vision_feature_select_strategy = mllm.config.vision_feature_select_strategy
            # 1. Extra the input embeddings
            inputs_embeds = mllm.get_input_embeddings()(input_ids)
            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = mllm.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {mllm.config.vision_feature_select_strategy}"
                    )
                image_features = mllm.multi_modal_projector(selected_image_feature)
                inputs_embeds, attention_mask, labels, position_ids = mllm._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, None
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, mllm.config.ignore_index).to(torch.long)
            inpt = {'attention_mask': attention_mask, 'inputs_embeds': inputs_embeds, 'position_ids': position_ids}
            return inpt
        inpt = self.processor(texts, imgs, return_tensors = 'pt', padding = True)
        for k, v in inpt.items(): inpt[k] = v.to(self.device[0]) if hasattr(v, 'to') else v
        llm_inpt = get_llava_llm_inpt(self.model, inpt.input_ids, inpt.attention_mask, inpt.pixel_values)
        if imgs != None:
            img_begin = torch.where(inpt['input_ids'][0] == self.get_img_special_token_id())[0][0]
            img_end = img_begin + self.get_img_token_n()
            vt_range = [int(img_begin), int(img_end)]
        else:
            vt_range = None
        return llm_inpt, vt_range

    def get_llm_outpt(self, llm_inpt, vt_range = None):
        assert 'inputs_embeds' in llm_inpt.keys()
        sig = inspect.signature(self.model.language_model.forward)
        llm_inpt = {k: v for k, v in llm_inpt.items() if k in sig.parameters}
        outpt = self.model.language_model(**llm_inpt, use_cache = False)
        return outpt

    def get_img_special_token_str(self):
        return '<image>'

    def get_img_special_token_id(self):
        return self.model.config.image_token_index
        
    def get_img_token_n(self):
        return (self.model.config.vision_config.image_size//self.model.config.vision_config.patch_size)**2 

    def is_q_former_based(self):
        return False

