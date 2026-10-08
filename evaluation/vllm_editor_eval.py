from editor.vllm_editors.base import VLLMBaseEditor
from editor.vllms_for_edit.base import BaseVLLMForEdit
from dataset.vllm import BaseVLLMEditData
from typing import List, Dict, Union
from collections import defaultdict
from collections import Counter
from datetime import datetime
from copy import deepcopy
import torch, os, json
from tqdm import tqdm
from time import time
import numpy as np
import sys


def cfg_to_dict(cfg):
    new_cfg = cfg.__dict__
    for k, v in new_cfg.items():
        if hasattr(v, '__dict__'):
            new_cfg[k] = v.__dict__
    return new_cfg


def normalize_answer_for_eval(text):
    return " ".join(str(text).strip().lower().split())


def normalized_exact_match(prediction, target):
    return float(normalize_answer_for_eval(prediction) == normalize_answer_for_eval(target))


def normalized_token_accuracy(prediction, target, tokenizer):
    prediction = normalize_answer_for_eval(prediction)
    target = normalize_answer_for_eval(target)
    pred_ids = tokenizer(prediction, add_special_tokens=False).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    max_len = max(len(pred_ids), len(target_ids))
    if max_len == 0:
        return 1.0
    pred_counter = Counter(pred_ids)
    target_counter = Counter(target_ids)
    matched = sum((pred_counter & target_counter).values())
    return matched / max_len


class VLLMEditorEvaluation():
    def __init__(self, editor:VLLMBaseEditor, eval_data:BaseVLLMEditData, 
        evaluation_name = None, results_dir = 'eval_results') -> None:
        '''
        `results_dir` & `evaluation_name`: Used to create result directory.
            `evaluation_name` can be set as dataset name.
        '''
        self.editor = editor
        self.eval_data = eval_data
        editor_name, model_name = editor.name_of_editor_and_model()
        t = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        evaluation_name = evaluation_name if evaluation_name else t
        self.result_dir = os.path.join(results_dir, editor_name, model_name, evaluation_name)
        print('Evaluation results directory: ', self.result_dir)
        # self.eval_xyms = None

    def evaluate_single_edit(self):
        editor = self.editor
        print('Evaluating reliability, generality and locality for %s on %s with single editing.'
              %editor.name_of_editor_and_model())
        eval_data = deepcopy(self.eval_data.data_with_img)
        for ed in eval_data: # single edit, the number of requests must be 1.
            assert len(ed['requests']) == 1
        result_data = deepcopy(self.eval_data.data_with_img_path)
        tokenizer = editor.vllm.get_llm_tokenizer()
        pre_edit_prompts_imgs_target_to_xym = getattr(
            editor,
            '_orig_prompts_imgs_target_to_xym',
            editor.vllm.prompts_imgs_target_to_xym,
        )
        editor.restore_to_original_model()  
        results = [] 
        for rd, ed in zip(tqdm(result_data, 'Evaluating'), eval_data):
            rd['reliability'] = rd.pop('requests') 
            rd['reliability'][0]['target'] = rd['reliability'][0].pop('target_new')
            # predict before edit for locality data
            for loc_name in ed['locality'].keys():
                for rdl, edl in zip(rd['locality'][loc_name], ed['locality'][loc_name]):
                    (input_embeds, vt_range), label_ids, label_masks = pre_edit_prompts_imgs_target_to_xym(
                        [edl['prompt']], [edl['image']], [edl['target']])
                    logits = editor.vllm.get_llm_outpt(input_embeds, vt_range).logits
                    before_edit_ids = torch.softmax(logits, -1).argmax(-1)[:, -label_ids.shape[1]:] # [1, l2]
                    rdl['predict_before_edit'] = tokenizer.decode(label_ids[label_masks.to(label_ids.device).to(bool)])
                    edl['before_edit_ids'] = before_edit_ids
                    edl['before_edit_label_masks'] = label_masks
            # edit
            start_t = time()
            edr = ed['requests'][0]
            edr['generality'] = ed['generality']
            editor.edit_one_piece(edr)
            editor.materialize_pending_edits()
            rd['reliability'][0]['edit_time'] = time() - start_t
            # compute scores 
            rd = self.__get_results_after_edit__(editor.vllm, ed, rd)
            results.append(rd)
            # Restore to original model
            editor.restore_to_original_model()
        save_dir = os.path.join(self.result_dir, 'single_edit')
        # save results
        self.save_results(os.path.join(save_dir, 'results.json'), results)
        mean_results = self.get_mean_results(results)
        mean_results['sample_count'] = len(results)
        self.save_results(os.path.join(save_dir, 'mean_results.json'), mean_results)
        return results

    def evaluate_sequential_edit(self, cfg, random = False, seed = None):
        edit_n = cfg.sequential_edit_n
        if edit_n is None or edit_n >= len(self.eval_data.data_with_img):
            edit_n = len(self.eval_data.data_with_img)
        online_sequential_eval = cfg.online_sequential_eval
        mode_dir = 'sequential_edit_%s%s' % (edit_n, '_online' if online_sequential_eval else '')
        run_dir = 'run_%s' % datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        experiment_name_postfix = str(getattr(cfg, 'experiment_name_postfix', '')).strip()
        if experiment_name_postfix != '':
            run_dir = '%s_%s' % (run_dir, experiment_name_postfix)
        save_dir = os.path.join(
            self.result_dir,
            (cfg.data_filename if cfg.data_filename else ''),
            (cfg.split if cfg.split else ''),
            ('data_size_%s'%cfg.data_sample_n),
            mode_dir,
            run_dir,
        )
        os.makedirs(save_dir, exist_ok=True)
        editor = self.editor
        print('Evaluating reliability, generality and locality for %s on %s with sequential editing %s%s.'
              %(*editor.name_of_editor_and_model(), edit_n, ' (online)' if online_sequential_eval else ' (reset between sessions)'))
        # preprocess data for sequential editing evaluation
        def split_data(data): 
            splited_data = []
            splited_data_ns = []
            now_split = []
            now_split_edit_n = 0
            for d in data:
                now_split.append(d)
                now_split_edit_n += len(d['requests'])
                if now_split_edit_n >= edit_n:
                    splited_data.append(now_split)
                    splited_data_ns.append(now_split_edit_n)
                    now_split = []
                    now_split_edit_n = 0
            if now_split_edit_n > 0:
                splited_data.append(now_split)
                splited_data_ns.append(now_split_edit_n)
            return splited_data, splited_data_ns
        eval_data = deepcopy(self.eval_data.data_with_img)
        result_data = deepcopy(self.eval_data.data_with_img_path)
        if random:
            seed = seed if seed != None else np.random.randint(1, 999999)
            np.random.default_rng(seed).shuffle(eval_data)
            np.random.default_rng(seed).shuffle(result_data)
        eval_data, eval_data_ns = split_data(eval_data)
        result_data, _ = split_data(result_data)
        # evaluate
        tokenizer = editor.vllm.get_llm_tokenizer()
        pre_edit_prompts_imgs_target_to_xym = getattr(
            editor,
            '_orig_prompts_imgs_target_to_xym',
            editor.vllm.prompts_imgs_target_to_xym,
        )
        editor.restore_to_original_model()
        results = []
        cur_session = -1
        for split_rd, split_ed in zip(tqdm(result_data, 'Evaluating'), eval_data):
            cur_session += 1
            split_res = []
            # prepare and locality
            for rd, ed in zip(tqdm(split_rd, 'Preparing', leave = False), split_ed):
                rd['reliability'] = rd.pop('requests') 
                for r in rd['reliability']:
                    r['target'] = r.pop('target_new')
                for loc_name in ed['locality'].keys(): # predict before edit for locality data
                    for rdl, edl in zip(rd['locality'][loc_name], ed['locality'][loc_name]):
                        (input_embeds, vt_range), label_ids, label_masks = pre_edit_prompts_imgs_target_to_xym(
                            [edl['prompt']], [edl['image']], [edl['target']])
                        logits = editor.vllm.get_llm_outpt(input_embeds, vt_range).logits
                        before_edit_ids = torch.softmax(logits, -1).argmax(-1)[:, -label_ids.shape[1]:] # [1, l2]
                        rdl['predict_before_edit'] = tokenizer.decode(before_edit_ids[label_masks.to(before_edit_ids.device).to(bool)])
                        edl['before_edit_ids'] = before_edit_ids
                        edl['before_edit_label_masks'] = label_masks
            # edit
            for rd, ed in zip(tqdm(split_rd, 'Editing', leave = False), split_ed): # edit
                for rdr, edr in zip(rd['reliability'], ed['requests']):
                    start_t = time()
                    edr['generality'] = ed['generality']
                    editor.edit_one_piece(edr)
                    rdr['edit_time'] = time() - start_t
            editor.materialize_pending_edits()
            # save ckpt
            if cfg.save_editor_checkpoint:
                checkpoint_save_path = os.path.join(save_dir, f"checkpoint_session_{cur_session}.pt")
                editor.save_ckpt_eval(cfg, checkpoint_save_path)
            # test
            for rd, ed in zip(tqdm(split_rd, 'Testing', leave = False), split_ed): # compute scores 
                rd = self.__get_results_after_edit__(editor.vllm, ed, rd)
                # rd = self.__get_results_after_edit_no_teacher_forcing__(editor.vllm, ed, rd)
                split_res.append(rd)
            if not online_sequential_eval:
                editor.restore_to_original_model()
            results.append(split_res)
        editor.restore_to_original_model()
        # save results
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump({'eval_config': cfg_to_dict(cfg), 'editor_config': cfg_to_dict(editor.cfg)}, f, indent=4)
        self.save_results(os.path.join(save_dir, '%sresults.json'%('seed_%s_'%seed if random else '')), results)
        print("Saved results to %s"%os.path.join(save_dir, '%sresults.json'%('seed_%s_'%seed if random else '')))
        split_mean = [self.get_mean_results(sr) for sr in results]
        cumulative_n = 0
        for mr, n in zip(split_mean, eval_data_ns):
            mr['sequential_edit_n'] = n
            if online_sequential_eval:
                cumulative_n += n
                mr['cumulative_edit_n'] = cumulative_n
        total_mean = self.get_mean_results([r for sr in results for r in sr])
        total_mean['total_edit_n'] = sum(eval_data_ns)
        mean_results = {"total_mean": total_mean, "split_mean": split_mean}
        self.save_results(os.path.join(save_dir, '%smean_results.json'%('seed_%s_'%seed if random else '')), mean_results)
        print("Saved mean results to %s"%os.path.join(save_dir, '%smean_results.json'%('seed_%s_'%seed if random else '')))
        return results

    def __get_results_after_edit__(self, vllm:BaseVLLMForEdit, ed, rd):
        def get_eval_xym(prompt, image, target, return_extra_info, image_path=None):
            (x, vt_range), y, m, extra_info = vllm.prompts_imgs_target_to_xym([prompt], [image], [target], return_extra_info=return_extra_info, image_paths=[image_path] if image_path else None)
            x['query_triple'] = (prompt, image, target)
            x['query_range'] = (0, x['inputs_embeds'].shape[1] - m.shape[1] + 1)
            return (x, vt_range), y, m, extra_info
        def accuracy_and_prediction(input_embeds, vt_range, label_ids, label_masks):
            # label_ids/label_masks: [1, l2]
            assert len(label_ids) == 1 and len(label_masks) == 1
            logits = vllm.get_llm_outpt(input_embeds, vt_range).logits # [1,l1,d]
            pre_y = torch.softmax(logits, -1).argmax(-1) # [1, l1]
            pre_y = pre_y[:, -label_ids.shape[1]:] # [1, l2]
            label_ids_for_acc = label_ids.to(pre_y.device)
            label_masks_for_acc = label_masks.to(pre_y.device)
            acc = ((pre_y == label_ids_for_acc) * label_masks_for_acc).sum()/label_masks_for_acc.sum()
            return float(acc), pre_y
        def decode_masked(ids, masks):
            return tokenizer.decode(ids[masks.to(ids.device).to(bool)])
        tokenizer = vllm.get_llm_tokenizer()
        # reliability
        for rdr, edr in zip(rd['reliability'], ed['requests']):
            (input_embeds, vt_range), label_ids, label_masks, extra_info = get_eval_xym(
                    edr['prompt'], edr['image'], edr['target_new'], True, edr.get('image_path'))
            acc, pre_y = accuracy_and_prediction(input_embeds, vt_range, label_ids, label_masks)
            pred_text = decode_masked(pre_y, label_masks)
            target_text = decode_masked(label_ids, label_masks)
            rdr['predict_after_edit'] = pred_text
            rdr['acc'] = acc
            rdr['norm_acc'] = normalized_token_accuracy(pred_text, target_text, tokenizer)
            rdr['norm_em'] = normalized_exact_match(pred_text, target_text)
            rdr['extra_info'] = extra_info
        # generality
        for gen_name in ed['generality']:
            for rdg, edg in zip(rd['generality'][gen_name], ed['generality'][gen_name]):
                (input_embeds, vt_range), label_ids, label_masks, extra_info = get_eval_xym(
                    edg['prompt'], edg['image'], edg['target'], True, edg.get('image_path'))
                acc, pre_y = accuracy_and_prediction(input_embeds, vt_range, label_ids, label_masks)
                pred_text = decode_masked(pre_y, label_masks)
                target_text = decode_masked(label_ids, label_masks)
                rdg['predict_after_edit'] = pred_text
                rdg['acc'] = acc
                rdg['norm_acc'] = normalized_token_accuracy(pred_text, target_text, tokenizer)
                rdg['norm_em'] = normalized_exact_match(pred_text, target_text)
                rdg['extra_info'] = extra_info
        # locality
        for loc_name in ed['locality']:
            for rdl, edl in zip(rd['locality'][loc_name], ed['locality'][loc_name]):
                (input_embeds, vt_range), _, _, extra_info = get_eval_xym(
                    edl['prompt'], edl['image'], edl['target'], True, edl.get('image_path'))
                label_masks = edl['before_edit_label_masks']
                acc, pre_y = accuracy_and_prediction(input_embeds, vt_range, edl['before_edit_ids'], label_masks)
                pred_text = decode_masked(pre_y, label_masks)
                target_text = decode_masked(edl['before_edit_ids'], label_masks)
                rdl['predict_after_edit'] = pred_text
                rdl['acc'] = acc
                rdl['norm_acc'] = normalized_token_accuracy(pred_text, target_text, tokenizer)
                rdl['norm_em'] = normalized_exact_match(pred_text, target_text)
                rdl['extra_info'] = extra_info
        return rd
    
    def __get_results_after_edit_no_teacher_forcing__(self, vllm:BaseVLLMForEdit, ed, rd):
        def normalize_target(prompt, target, space=False):
            if len(target) == 0:
                return target
            needs_space = (prompt[-1] not in [' ', '\n']) if len(prompt) > 0 else True
            if space:
                if needs_space and target[0] not in [' ', '\n']:
                    return ' ' + target
            return target
        def get_eval_input_target(prompt, image, target):
            llm_inpt, vt_range = vllm.get_llm_input_embeds([prompt], [image])
            llm_inpt['query_triple'] = (prompt, image, target)
            llm_inpt['query_range'] = (0, llm_inpt['inputs_embeds'].shape[1])
            target_text = normalize_target(prompt, target)
            tokenized = tokenizer([target_text], return_tensors='pt', padding=True).input_ids.to(vllm.device[-1])
            if tokenized.shape[1] > 1:
                label_ids = tokenized[:, 1:2]
            else:
                label_ids = torch.full((tokenized.shape[0], 1), pad_token_id, dtype=tokenized.dtype, device=vllm.device[-1])
            return (llm_inpt, vt_range), label_ids
        def accuracy_and_prediction(input_embeds, vt_range, label_ids):
            assert label_ids.shape == (1, 1)
            logits = vllm.get_llm_outpt(input_embeds, vt_range).logits
            next_token = logits[:, -1:, :].argmax(-1)
            acc = float((next_token == label_ids).float().mean())
            return acc, next_token
        tokenizer = vllm.get_llm_tokenizer()
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        # reliability
        for rdr, edr in zip(rd['reliability'], ed['requests']):
            (input_embeds, vt_range), label_ids = get_eval_input_target(
                    edr['prompt'], edr['image'], edr['target_new'])
            acc, pre_y = accuracy_and_prediction(input_embeds, vt_range, label_ids)
            pred_text = tokenizer.batch_decode(pre_y)[0]
            target_text = tokenizer.batch_decode(label_ids)[0]
            rdr['predict_after_edit'] = tokenizer.batch_decode(pre_y)
            rdr['acc'] = acc
            rdr['norm_acc'] = normalized_token_accuracy(pred_text, target_text, tokenizer)
            rdr['norm_em'] = normalized_exact_match(pred_text, target_text)
        # generality
        for gen_name in ed['generality']:
            for rdg, edg in zip(rd['generality'][gen_name], ed['generality'][gen_name]):
                (input_embeds, vt_range), label_ids = get_eval_input_target(
                    edg['prompt'], edg['image'], edg['target'])
                acc, pre_y = accuracy_and_prediction(input_embeds, vt_range, label_ids)
                pred_text = tokenizer.batch_decode(pre_y)[0]
                target_text = tokenizer.batch_decode(label_ids)[0]
                rdg['predict_after_edit'] = tokenizer.batch_decode(pre_y)
                rdg['acc'] = acc
                rdg['norm_acc'] = normalized_token_accuracy(pred_text, target_text, tokenizer)
                rdg['norm_em'] = normalized_exact_match(pred_text, target_text)
        # locality
        for loc_name in ed['locality']:
            for rdl, edl in zip(rd['locality'][loc_name], ed['locality'][loc_name]):
                (input_embeds, vt_range), label_ids = get_eval_input_target(
                    edl['prompt'], edl['image'], edl['target'])
                acc, pre_y = accuracy_and_prediction(input_embeds, vt_range, label_ids)
                pred_text = tokenizer.batch_decode(pre_y)[0]
                target_text = tokenizer.batch_decode(label_ids)[0]
                rdl['predict_after_edit'] = tokenizer.batch_decode(pre_y)
                rdl['acc'] = acc
                rdl['norm_acc'] = normalized_token_accuracy(pred_text, target_text, tokenizer)
                rdl['norm_em'] = normalized_exact_match(pred_text, target_text)
        return rd

    def get_mean_results(self, results:List[Dict]):
        """Get numbers from a result: {
            "reliability": [
                {"acc": float, "edit_time": float}, 
                {"acc": float, "edit_time": float}, ...]
            "generality": {
                sub_metric_1: [{"acc": float}, {"acc": float}, ...], 
                sub_metric_2: [{"acc": float}, {"acc": float}, ...], ...}
            "locality": {
                sub_metric_1: [{"acc": float}, {"acc": float}, ...], 
                sub_metric_2: [{"acc": float}, {"acc": float}, ...], ...}
        }
        """
        mean_res = {"reliability": {}, "generality": {}, "locality": {}}
        # sum values
        for r in results:
            for rr in r['reliability']:
                for value_name, value in rr.items():
                    if isinstance(value, (int, float)):
                        if value_name not in mean_res['reliability']:
                            mean_res['reliability'][value_name] = [0, 0]
                        mean_res['reliability'][value_name][0] += value
                        mean_res['reliability'][value_name][1] += 1
            for sub_metric in r['generality'].keys():
                if sub_metric not in mean_res['generality']:
                    mean_res['generality'][sub_metric] = {}
                for sub_res in r['generality'][sub_metric]:
                    for value_name, value in sub_res.items():
                        if isinstance(value, (int, float)):
                            if value_name not in mean_res['generality'][sub_metric]:
                                mean_res['generality'][sub_metric][value_name] = [0, 0]
                            mean_res['generality'][sub_metric][value_name][0] += value
                            mean_res['generality'][sub_metric][value_name][1] += 1
            for sub_metric in r['locality'].keys():
                if sub_metric not in mean_res['locality']:
                    mean_res['locality'][sub_metric] = {}
                for sub_res in r['locality'][sub_metric]:
                    for value_name, value in sub_res.items():
                        if isinstance(value, (int, float)):
                            if value_name not in mean_res['locality'][sub_metric]:
                                mean_res['locality'][sub_metric][value_name] = [0, 0]
                            mean_res['locality'][sub_metric][value_name][0] += value
                            mean_res['locality'][sub_metric][value_name][1] += 1
        # compute mean results
        for value_name, value in mean_res['reliability'].items():
            mean_res['reliability'][value_name] = value[0] / value[1]
        for sub_metric in mean_res['generality'].keys():
            for value_name, value in mean_res['generality'][sub_metric].items():
                mean_res['generality'][sub_metric][value_name] = value[0] / value[1]
        for sub_metric in mean_res['locality'].keys():
            for value_name, value in mean_res['locality'][sub_metric].items():
                mean_res['locality'][sub_metric][value_name] = value[0] / value[1]
        return mean_res

    def save_results(self, save_path:str, results:Dict, decimal = 4):
        def set_decimal(r):
            if isinstance(r, list):
                for i in range(len(r)):
                    r[i] = set_decimal(r[i])
            elif isinstance(r, dict) or isinstance(r, defaultdict):
                for k in r.keys():
                    r[k] = set_decimal(r[k])
            elif isinstance(r, float):
                r = round(r, decimal)
            return r
        res = deepcopy(results)
        res = set_decimal(res)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(os.path.join(save_path), 'w') as f:
            json.dump(res, f, indent = 4)
