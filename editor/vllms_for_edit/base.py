from typing import Dict, List, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence
from ..nethook import TraceDict, Trace
from PIL.Image import Image as ImageClass
from transformers import  AutoTokenizer
from PIL.Image import Image as PILImage
from abc import ABC, abstractmethod
from torch import nn
import os, torch
import numpy as np


def move_to_device(data, device):
    '''Move list and dictionary nested PyTorch tensors to a specific device.'''
    if isinstance(data, (torch.Tensor, nn.Module)):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple([move_to_device(item, device) for item in data])
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (int, float, str, bool, type(None), np.integer, np.floating)):
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def get_multiple_gpus_for_vllm(vllm_path, devices):
    if len(devices) == 1:
        if vllm_path == "llava-hf/llava-1.5-7b-hf":
            num_llm_layers = 32
            device_map = {
                "vision_tower": devices[0],
                "multi_modal_projector": devices[0],
                "language_model.model.embed_tokens": devices[0],
                "language_model.model.norm": devices[0],
                "language_model.model.rotary_emb": devices[0],
                "language_model.lm_head": devices[0],
            }
            for i in range(num_llm_layers):
                key = f"language_model.model.layers.{i}"
                device_map[key] = devices[0]
    elif len(devices) == 2:
        if vllm_path == "llava-hf/llava-1.5-7b-hf":
            num_llm_layers = 32
            first_gpu_last_layer = int(os.environ.get("LLAVA_FIRST_GPU_LAST_LAYER", "11"))
            device_map = {
                "vision_tower": devices[0],
                "multi_modal_projector": devices[0],
                "language_model.model.embed_tokens": devices[0],
                "language_model.model.norm": devices[1],
                "language_model.model.rotary_emb": devices[1],
                "language_model.lm_head": devices[1],
            }
            for i in range(num_llm_layers):
                key = f"language_model.model.layers.{i}"
                device_map[key] = devices[0] if (i <= first_gpu_last_layer) else devices[1]
    elif len(devices) == 4:
        if vllm_path == "llava-hf/llava-1.5-7b-hf":
            num_llm_layers = 32
            # Keep GPU0 light because RECIPE also places RoBERTa/prompt modules there.
            layer_ends = [int(x) for x in os.environ.get("LLAVA_FOUR_GPU_LAYER_ENDS", "3,13,23").split(",")]
            if len(layer_ends) != 3:
                raise ValueError("LLAVA_FOUR_GPU_LAYER_ENDS must contain three comma-separated layer ids.")
            device_map = {
                "vision_tower": devices[0],
                "multi_modal_projector": devices[0],
                "language_model.model.embed_tokens": devices[0],
                "language_model.model.rotary_emb": devices[0],
                "language_model.model.norm": devices[3],
                "language_model.lm_head": devices[3],
            }
            for i in range(num_llm_layers):
                key = f"language_model.model.layers.{i}"
                if i <= layer_ends[0]:
                    device_map[key] = devices[0]
                elif i <= layer_ends[1]:
                    device_map[key] = devices[1]
                elif i <= layer_ends[2]:
                    device_map[key] = devices[2]
                else:
                    device_map[key] = devices[3]
    else:
        raise ValueError(f"Unsupported number of devices: {len(devices)}")
    return device_map


def set_tokenizer_pad_id(tokenizer:AutoTokenizer, padding_side = 'right'):
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print('Set [pad_token] as [eos_token].')
    print('Padding side is set as "%s".'%padding_side)
    tokenizer.padding_side = padding_side

############################################################################
########################## VLLM Wrap Class #################################
############################################################################
class BaseVLLMForEdit(ABC):
    '''
    A wrap of VLLM that first converts both text and image into embedded 
    representations of the language model, and then achieves the subsequent inference.
    ''' 
    def __init__(self, model:nn.Module, device:List[str], auto_add_img_special_token:bool) -> None:
        '''`auto_add_img_special_token`: if available, whether add image special 
            token automatically for input prompts.'''
        super().__init__() 
        self.model = model
        self.device = device
        self.auto_add_img_special_token = auto_add_img_special_token
        set_tokenizer_pad_id(self.get_llm_tokenizer(), padding_side = 'right')
        self.get_llm_input_embeds = self.__get_llm_input_embeds_wrap__(self.get_llm_input_embeds)

    def __get_llm_input_embeds_wrap__(self, get_llm_input_embeds):
        def wrapped_get_llm_input(texts:List[str], imgs:Optional[List[ImageClass]] = None):
            ''' 
            1. If `self.auto_add_img_special_token` is True, automatically add image 
                special token string into texts.
            2. Check if the texts and images are legal.'''
            # only support types
            if not isinstance(imgs, (list, type(None))) or not isinstance(texts, list): 
                raise BaseException('Not support type.')
            if isinstance(imgs, list) and all(i == None for i in imgs):
                imgs = None 
            ist_str = self.get_img_special_token_str()
            # auto add image special token for texts
            if self.auto_add_img_special_token and imgs != None and ist_str != None:
                texts = [ist_str + '\n' + t if t.find(ist_str) == -1 else t for t in texts]
            if imgs == None:
                # if `imgs` is None and has image special token, 
                # texts must not include image special token.
                if ist_str != None:
                    for t in texts:
                        if t.find(ist_str) != -1:
                            raise BaseException('`imgs` is None but found special image token in `texts`.')
            else:
                # if `imgs` is not None, number of images and texts must be same.
                if len(texts) != len(imgs):
                    raise BaseException('Number of texts (n = %s) and images (n = %s) not matched.'%(len(texts), len(imgs)))
                # if `imgs` is not None and image special token is not None, 
                # the positions of image special token in texts must be same.
                if ist_str != None:
                    img_token_str_begin = texts[0].find(ist_str) 
                    for t in texts:
                        if t.count(ist_str) != 1:
                            raise BaseException('One image must correspond to one text.')
                        if t[:img_token_str_begin] != texts[0][:img_token_str_begin]:
                            raise BaseException('Special image token with different prefixes is not supported')
            return get_llm_input_embeds(texts, imgs)
        return wrapped_get_llm_input

    def prompts_imgs_target_to_xym(self, prompts:List[str], imgs:List[PILImage],
                                   targets:List[str], return_extra_info:bool = False, **kwargs):
        '''
        Assume batch_size is `len(prompts)`, equals `len(imgs)`, equals `len(targets)`.
        return (type, dtype, shape):
            1. `(input_embeds, vt_range)`, output from `self.get_llm_input_embeds`:
                (torch.Tensor, float, [batch_size, l_total, d])
            2. `label_ids`, predict ids:
                (torch.Tensor, Long, [batch_size, l_short])
            3. `label_masks`, mask of predict ids for training:
                (torch.Tensor, Long, [batch_size, l_short])
        4. (if `return_extra_info=True`) extra_info dict for method-specific metadata.
        The `l_total` is the length of all input tokens, the `l_short` is the length of tokens
            used to compute loss.
        '''
        targets = [' ' + t if p[-1] not in [' ', '\n'] and t[0] not in [' ', '\n'] else t 
                   for p, t in zip(prompts, targets)]
        tokenizer = self.get_llm_tokenizer()
        input_strs, label_ids, label_masks = [], [], []
        min_prompt_tok_n = 999
        for p, t in zip(prompts, targets):
            inpt_str = p + t
            input_strs.append(inpt_str)
            label_tok = tokenizer(inpt_str, return_tensors = 'pt', padding = True)['input_ids'][0]
            label_tok = torch.roll(label_tok, -1, 0)
            label_ids.append(label_tok)
            mask = torch.zeros_like(label_tok)
            prompt_tok = tokenizer(p, return_tensors = 'pt', padding = True)['input_ids'][0]
            if min_prompt_tok_n > len(prompt_tok):
                min_prompt_tok_n = len(prompt_tok)
            mask[len(prompt_tok)-1:-1] += 1
            label_masks.append(mask)
        input_embeds, vt_range = self.get_llm_input_embeds(input_strs, imgs)
        label_ids = pad_sequence(label_ids, True, tokenizer.pad_token_id).to(self.device[-1])[:, min_prompt_tok_n-1:]
        label_masks = pad_sequence(label_masks, True, 0).to(self.device[-1])[:, min_prompt_tok_n-1:]
        if return_extra_info:
            return (input_embeds, vt_range), label_ids, label_masks, {}
        return (input_embeds, vt_range), label_ids, label_masks

    def label_loss(self, logits, label_ids, label_masks, average = True):
        # logits: [batch_size, total_l, d], label_ids/label_masks: [batch_size, short_l]
        label_ids = move_to_device(label_ids, logits.device)
        label_masks = move_to_device(label_masks, logits.device)
        # logits = logits[:, -label_ids.shape[1]:]
        # Guard against label_ids longer than logits (can happen in VLMs when
        # image token expansion makes the embedding shorter than the label sequence
        # for certain inputs).
        short_l = min(label_ids.shape[1], logits.shape[1])
        if short_l < label_ids.shape[1]:
            label_ids = label_ids[:, -short_l:]
            label_masks = label_masks[:, -short_l:]
        logits = logits[:, -short_l:]
        log_pre_p = torch.log_softmax(logits, -1)
        log_pre_p = log_pre_p.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1) # [batch, short_l]
        loss = -(log_pre_p * label_masks).sum()
        if average:
            loss = loss / label_masks.sum() 
        return loss
    
    def logit_KL_loss(self, logits1, logits2, label_masks, average = True):
        # logits1/logits2: [batch, total_l, voc_size], label_masks: [batch, short_l]
        logits2 = move_to_device(logits2, logits1.device)
        label_masks = move_to_device(label_masks, logits1.device)
        logits1 = logits1[:, -label_masks.shape[1]:]
        logits2 = logits2[:, -label_masks.shape[1]:]
        log_p1 = torch.log_softmax(logits1, -1)
        log_p2 = torch.log_softmax(logits2, -1)
        p1 = torch.softmax(logits1, 2)
        kl_loss = (p1 * (log_p1 - log_p2)).sum(2) # [batch, max_len]
        loss = (kl_loss * label_masks).sum()
        if average:
            loss = loss / label_masks.sum() 
        return loss

    def set_device(self, device):
        self.device = device
        # self.model.to(device)

    def get_mid_module_inpt(self, input_embeds, vt_range, mid_module_path, 
                            get_first_if_tuple = True):
        with torch.no_grad(), Trace(self.model, mid_module_path, retain_input = True, 
                with_kwargs = False, stop = True, clone = False) as t:
            self.get_llm_outpt(input_embeds, vt_range)
        if isinstance(t.input, torch.Tensor):
            return t.input
        elif isinstance(t.input, (list, tuple)):
            if get_first_if_tuple:
                return t.input[0] # args[0] is the middle layer representation
            return t.input # args[0] is the middle layer representation
        raise "Unknown Type!"

    def get_mid_module_outpt(self, input_embeds, vt_range, mid_module_path,
                             get_first_if_tuple = True):
        with torch.no_grad(), Trace(self.model, mid_module_path, retain_output = True, 
                with_kwargs = False, stop = True, clone = False) as t:
            self.get_llm_outpt(input_embeds, vt_range)
        if isinstance(t.output, torch.Tensor):
            return t.output # output[0] is middle layer representation 
        elif isinstance(t.output, (list, tuple)):
            if get_first_if_tuple:
                return t.output[0] # output[0] is middle layer representation 
            return t.output # output[0] is middle layer representation 
        raise "Unknown Type!"
    
    def forward_from_mid_layer(self, llm_inpt, vt_range, mid_layer_inpt:torch.Tensor, 
                             llm_layer_tmp:str, mid_inpt_layer_i:int):
        ''' Use `mid_layer_inpt` to inference from LLM's middle layer for speed up inference.
        Note! Only support basic layer, and do not support other modules Att and FFN.
        The shape of hidden representation of (llm_inpt, vt_range) must 
        be same with `mid_layer_inpt`. 
        `llm_layer_tmp`: the layer name template. 
        `mid_inpt_layer_i`: the index of the middle layer to begin inference.
        '''
        def skip_layer(*args, **kargs):
            return [None]*2
        def mid_inpt_embeds(inpt, layer):
            args, kargs = td[llm_layer_tmp.format(0)].input
            args = (mid_layer_inpt,)
            return args, kargs
        skip_layers = [llm_layer_tmp.format(i) for i in range(mid_inpt_layer_i)]
        inpt_layer = [llm_layer_tmp.format(mid_inpt_layer_i)]
        with TraceDict(self.model, [llm_layer_tmp.format(0)], retain_input=True
            ) as td, TraceDict(self.model, skip_layers, layer_func_replace=skip_layer
            ), TraceDict(self.model, inpt_layer, edit_input=mid_inpt_embeds): 
            outpt = self.get_llm_outpt(llm_inpt, vt_range)
        return outpt

    def find_closest_tokens(self, embeddings:torch.Tensor, embedding_matrix:torch.Tensor, top_k = 1):
        # embeddings: [b, l, d]; embedding_matrix: [v, d]
        # Normalization
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
        embedding_matrix_norm = embedding_matrix / embedding_matrix.norm(dim=1, keepdim=True)
        # Similarity
        cosine_sim = torch.matmul(embeddings_norm, embedding_matrix_norm.T)
        top_k_indices = torch.topk(cosine_sim, top_k, dim=-1)
        toks, sim = top_k_indices.indices, top_k_indices.values
        return toks, sim

    @abstractmethod
    def get_llm_tokenizer(self)->AutoTokenizer:
        '''return the tokenizer of the llm in vllm.'''

    @abstractmethod
    def get_llm_input_embeds(self, texts:List[str], imgs:Optional[List[ImageClass]] = None)->Tuple[torch.Tensor, Tuple[int, int]]:
        '''
        `texts`: Input texts, only support one or no image special token.
        `imgs`: Input images, one image must correspond to only one text.
        return input_embeds, vt_range
            `input_embeds`: Input embeddings of the llm in the vllm.
            `vt_range`: Range of vision token in `inputs_embeds`. '''

    @abstractmethod
    def get_llm_outpt(self, input_embeds, vt_range = None):
        '''Only support one image in one text. 
        `input_embeds`, `vt_range`: from `self.get_llm_input_embeds`
        return: 
            output of the llm including `logits`
        '''
 
    @abstractmethod
    def get_img_special_token_str(self)->str:
        '''String of image special token.'''
        
    @abstractmethod
    def get_img_special_token_id(self)->int:
        '''Id of image special token.'''
        
    @abstractmethod
    def get_img_token_n(self)->int:
        '''Tokens number of one image transformed into word embedding space.'''
    
    @abstractmethod
    def is_q_former_based(self)->bool:
        '''Returns whether the model is based on Q-former.'''
