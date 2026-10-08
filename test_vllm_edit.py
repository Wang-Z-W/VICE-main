from utils import get_full_model_name, load_vllm_editor
from evaluation.vllm_editor_eval import VLLMEditorEvaluation
from utils.GLOBAL import ROOT_PATH
from dataset.vllm import EVQA, EIC, VLKEB, WaterBird, WaterBirdV2
import os, argparse, sys

_VICE_CACHE_CFG = {
    "EVQA": {
        "cache_dir": "data/easy-edit-mm/vqa",
        "diverse_prefix": "vqa",
        "data_cls": EVQA,
        "eval_json": "data/easy-edit-mm/vqa/vqa_eval.json",
        "img_dir": "data/easy-edit-mm/images",
    },
    "EIC": {
        "cache_dir": "data/easy-edit-mm/caption",
        "diverse_prefix": "caption",
        "data_cls": EIC,
        "eval_json": "data/easy-edit-mm/caption/caption_eval_edit.json",
        "img_dir": "data/easy-edit-mm/images",
    },
    "VLKEB": {
        "cache_dir": "data/VLKEB",
        "diverse_prefix": "VLKEB",
        "data_cls": VLKEB,
        "eval_json": "data/VLKEB/eval.json",
        "img_dir": "data/VLKEB/mmkb_images",
    },
    "WaterBird": {
        "cache_dir": "data/WaterBird",
        "diverse_prefix": "WaterBird",
        "data_cls": WaterBird,
        "eval_json": "data/WaterBird/edit_annotations_truelabel_balanced.json",
        "img_dir": "data/WaterBird",
    },
    "CUB_filtered": {
        "cache_dir": "data/CUB_filtered",
        "diverse_prefix": "CUB_filtered",
        "data_cls": WaterBird,
        "eval_json": "data/CUB_filtered/edit.json",
        "img_dir": "data/CUB_filtered",
    },
    "WaterBirdv2": {
        "cache_dir": "data/WaterBird-v2.0",
        "diverse_prefix": "WaterBird-v2.0",
        "data_cls": WaterBirdV2,
        "eval_json": "data/WaterBird-v2.0/edit_annotations.json",
        "img_dir": "",
    },
}

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

def get_attr():
    parser = argparse.ArgumentParser()
    parser.add_argument('-en', '--editor_name', type=str, help='Editor name: LiveEdit, FT_VL...', required=True)
    parser.add_argument('-mn', '--edit_model_name', type=str, help='Editing model name: llava...', required=True)
    parser.add_argument('-sen', '--sequential_edit_n', type=int, help='Edit number.', required=True)
    parser.add_argument(
        '-dnp',
        '--dataset_name_postfix',
        '-enp',
        '--eval_name_postfix',
        dest='dataset_name_postfix',
        type=str,
        default='',
        help='Postfix appended to the dataset/evaluation directory name.',
    )
    parser.add_argument(
        '-exp',
        '--experiment_name_postfix',
        type=str,
        default='',
        help='Postfix appended to the run directory name for this experiment.',
    )
    parser.add_argument('-dvc', '--device', type=str, help='CUDA device for editing.', required=True)
    parser.add_argument('-ckpt', '--editor_ckpt_path', type=str, default = None, help='For Editors that needs training.')
    parser.add_argument('-dn', '--data_name', type=str, required = True, help = 'Evaluating dataset, including EVQA, EIC.')
    parser.add_argument('-dsn', '--data_sample_n', type=int, default = None, help = 'Sample number for evaluation.')
    parser.add_argument('-cleanvqa', '--clean_vqa_eval', type=str2bool, default = False, help='Whether to evaluate on the clean EVQA subset.')
    # VICE-specific arguments
    parser.add_argument('-thr', '--sim_threshold', type=float, default = 0.75, help = 'Similarity threshold for selecting retrieved edits for VICE.')
    parser.add_argument('-kmeans', '--use_kmeans_for_diverse_embeddings', type=str2bool, default = True, help='Whether to use KMeans clustering for selecting diverse data to build diverse embeddings for VICE.')
    parser.add_argument('-usesum', '--use_image_summary', type=str2bool, default=True, help='Whether VICE uses image summary in diverse embeddings, memory construction, and retrieval.')
    parser.add_argument(
        '--image_summary_ablation',
        type=str,
        default='none',
        choices=[
            'none',
            'target_empty',
            'distractors_empty',
            'swap_fields',
            'field_swap',
            'no_target',
            'no_distractors',
            'no_evidence',
        ],
        help='VICE image-summary component ablation. target_empty/no_target empties Target Evidence; distractors_empty/no_distractors empties Distractors; swap_fields exchanges Target Evidence and Distractors.',
    )
    parser.add_argument('--scope_classifier_ckpt', type=str, default=None, help='Path to load the VICE scope projection checkpoint.')
    parser.add_argument('--no_icl', action='store_true', help='Disable in-context retrieval for VICE (ablation).')
    # WaterBird-specific arguments
    parser.add_argument('-dfn', '--data_filename', type=str, default = None, help = 'Filename for evaluation. Mainly for WaterBird dataset.')
    parser.add_argument('-spt', '--split', type=str, default = None, help = 'Data split. Mainly for WaterBird dataset.')

    parser.add_argument('-saveckpt', '--save_editor_checkpoint', action='store_true', help='Save editor checkpoint.')
    
    parser.add_argument(
        '-online',
        '--online_sequential_eval',
        type=str2bool,
        default=True,
        help='Keep edits accumulating across intervals and test every -sen edits.'
    )
    args = parser.parse_args()
    return args
    
def set_device(cfg):
    cfg.device = [f"cuda:{d}" for d in cfg.device.split(',')]

if __name__ == '__main__':
    cfg = get_attr()
    set_device(cfg)
    cfg.editor_name = cfg.editor_name.lower()
    cfg.edit_model_name = get_full_model_name(cfg.edit_model_name)
    cfg.evaluation_name = cfg.data_name.upper()
    dataset_name_postfix = str(cfg.dataset_name_postfix).strip().replace('/', '-').replace(' ', '_')
    cfg.dataset_name_postfix = dataset_name_postfix
    cfg.experiment_name_postfix = str(cfg.experiment_name_postfix).strip().replace('/', '-').replace(' ', '_')
    if cfg.dataset_name_postfix != '':
        cfg.evaluation_name = '%s-%s'%(cfg.evaluation_name, cfg.dataset_name_postfix)
    print(cfg)
    editor = load_vllm_editor(cfg, cfg.editor_name, cfg.edit_model_name, cfg.device, None, cfg.editor_ckpt_path, False)
    # load data
    _supported_vice_datasets = {"EVQA", "EIC", "VLKEB", "WaterBird", "CUB_filtered", "WaterBirdv2"}
    if cfg.data_name in _supported_vice_datasets:
        vcfg = _VICE_CACHE_CFG[cfg.data_name]
        DatasetCls = vcfg["data_cls"]
        cache_dir = os.path.join(ROOT_PATH, vcfg["cache_dir"])
        diverse_prefix = vcfg["diverse_prefix"]

        if cfg.data_name == 'EVQA' and cfg.clean_vqa_eval:
            data_path = os.path.join(ROOT_PATH, 'data/easy-edit-mm/vqa/vqa_eval_clean.json')
            img_root_dir = os.path.join(ROOT_PATH, vcfg["img_dir"])
            eval_data = DatasetCls(data_path, img_root_dir, cfg.data_sample_n)
        elif cfg.data_name == 'WaterBirdv2':
            data_path = os.path.join(ROOT_PATH, vcfg["eval_json"])
            eval_split = cfg.split if cfg.split else "test"
            img_root_dir = ""  # v2.0 paths are project-root-relative
            eval_data = DatasetCls(data_path, img_root_dir, eval_split, cfg.data_sample_n)
        elif cfg.data_name in ('WaterBird', 'CUB_filtered'):
            data_path = os.path.join(ROOT_PATH, vcfg["eval_json"]) if cfg.data_filename is None else os.path.join(ROOT_PATH, f'data/{cfg.data_name}/{cfg.data_filename}.json')
            if cfg.data_filename is None and cfg.split is None:
                data_path = os.path.join(ROOT_PATH, vcfg["eval_json"])
                eval_split = "test"
            else:
                eval_split = cfg.split if cfg.split else "test"
            img_root_dir = os.path.join(ROOT_PATH, vcfg["img_dir"])
            eval_data = DatasetCls(data_path, img_root_dir, eval_split, cfg.data_sample_n)
        else:
            data_path = os.path.join(ROOT_PATH, vcfg["eval_json"])
            img_root_dir = os.path.join(ROOT_PATH, vcfg["img_dir"])
            eval_data = DatasetCls(data_path, img_root_dir, cfg.data_sample_n)
        cfg.data_sample_n = len(eval_data.data_with_img)

        # VICE-specific data loading and preparation
        if cfg.editor_name.lower() == 'vice_vl':
            if getattr(cfg, 'no_icl', False):
                editor.no_icl = True
            if cfg.scope_classifier_ckpt:
                cls_ckpt = cfg.scope_classifier_ckpt
                if not os.path.isabs(cls_ckpt):
                    cls_ckpt = os.path.join(ROOT_PATH, cls_ckpt)
                if not os.path.exists(cls_ckpt):
                    raise FileNotFoundError(f"VICE scope classifier checkpoint not found: {cfg.scope_classifier_ckpt}")
                editor.load_classifier(cls_ckpt)
            if cfg.use_image_summary:
                train_sum = os.path.join(cache_dir, 'image_summaries_cache_train.json')
                edit_sum = os.path.join(cache_dir, 'image_summaries_cache_eval_edit.json')
                query_sum = os.path.join(cache_dir, 'image_summaries_cache_eval_query.json')
                if os.path.exists(train_sum):
                    editor.load_image_summary_cache(train_sum, edit_sum, query_sum)
                else:
                    print(f"Image summary cache not found at {train_sum}, skipping.")
            if cfg.use_kmeans_for_diverse_embeddings:
                div_path = os.path.join(cache_dir, f'{diverse_prefix}_train_diverse_kmeans.pt')
                if os.path.exists(div_path):
                    editor.load_diverse_embeddings(div_path)
                else:
                    print(f"Diverse embeddings not found at {div_path}, skipping (will use KMeans at runtime).")
            else:
                div_path = os.path.join(cache_dir, f'{diverse_prefix}_train_diverse.pt')
                if os.path.exists(div_path):
                    editor.load_diverse_embeddings(div_path)
                else:
                    print(f"Diverse embeddings not found at {div_path}, skipping.")
    elif cfg.data_name == 'SpuMNIST':
        from dataset.vllm import SpuMNIST
        data_path = os.path.join(ROOT_PATH, f'data/SpuMNIST/annotations/{cfg.data_filename}.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/SpuMNIST')
        eval_data = SpuMNIST(data_path, img_root_dir, cfg.data_sample_n)
        cfg.data_sample_n = len(eval_data.data_with_img)
        # Set diverse data for VICE if applicable (automatic KMeans clustering)
        if cfg.editor_name.lower() == 'vice_vl':
            raise NotImplementedError('VICE VL currently does not support SpuMNIST dataset.')
    else:
        eval_data = None
    # evaluate
    ev = VLLMEditorEvaluation(editor, eval_data, cfg.evaluation_name, 'eval_results')
    ev.evaluate_sequential_edit(cfg, False, None)
