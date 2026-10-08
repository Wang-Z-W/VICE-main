from utils.GLOBAL import ROOT_PATH
from utils import load_vllm_editor
import os, argparse, time


def get_attr():
    def parse_lkpt(value:str):
        if value.lower() == 'none':
            return None
        return value
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument('-en', '--editor_name', type=str, help='Editor name: LiveEdit, FT_VL...', required=True)
    parser.add_argument('-mn', '--edit_model_name', type=str, help='Editing model name: llava...', required=True)
    parser.add_argument('-dna', '--data_name', type=str, help = 'Train dataset, including EVQA, EIC, WaterBird.', required = True)
    parser.add_argument('-bs', '--batch_size', type=int, help = 'Train dataset sample number.', required = True)
    parser.add_argument('-dvc', '--device', type=str, help='CUDA device for editing.', required=True)
    # other settings
    parser.add_argument('-dfn', '--data_file_name', type=str, default=None, help = 'Train dataset file name.')
    parser.add_argument('-spt', '--split', type=str, default=None, help = 'Train dataset split.')
    parser.add_argument('-dn', '--data_n', type=int, default=None, help = 'Train dataset sample number.')
    parser.add_argument('-lkpt', '--load_ckpt_path', type=parse_lkpt, default = None, help='For Editors that needs training.')
    parser.add_argument(
        '-edvc', '--extra_devices', type=str, required=True,
        help='CUDA devices for the separate LiveEdit data-processing model.',
    )
    parser.add_argument('-eps', '--epochs', type=int, default=1000, help = 'Train epochs.')
    parser.add_argument('-tnp', '--train_name_prefix', type=str, default=None, help = 'Train name prefix.')
    parser.add_argument('-sci', '--save_ckpt_per_i', type=int, default=1000, help = 'Save checkpoint per iteraions.')
    parser.add_argument('-lpi', '--log_per_i', type=int, default=1, help = 'Log per iteraions.')
    parser.add_argument('-ea', '--ema_alpha', type=float, default=0.1, help = 'EMA loss alpha.')
    parser.add_argument('-rs', '--random_seed', type=int, default=None, help = 'Random seed.')
    parser.add_argument('-dbs', '--data_buffer_size', type=int, default=4, help = 'Buffer size of data generator.')
    args = parser.parse_args()
    return args


class cfg:
    editor_name = 'LiveEdit'
    edit_model_name = 'blip2'
    data_name = 'EVQA'
    batch_size = 4
    device = 'cuda:0'
    data_n = None
    load_ckpt_path = None
    extra_devices = [1]
    epochs = 1000
    train_name_prefix = 'testttt'
    save_ckpt_per_i = 1000
    log_per_i = 1
    ema_alpha = 0.1
    random_seed = 1
    data_buffer_size = 4
    
    
def set_device(cfg):
    cfg.device = [f"cuda:{d}" for d in cfg.device.split(',')]
    cfg.extra_devices = [f"cuda:{d}" for d in cfg.extra_devices.split(',')]


def resolve_random_seed(cfg):
    if cfg.random_seed is None:
        cfg.random_seed = int(time.time() * 10000) % 99999999


def get_train_name(cfg):
    data_n = 'all' if cfg.data_n is None else cfg.data_n
    dn_bs_dir = f'dn={data_n}_bs={cfg.batch_size}'
    seed_dir = f'seed={cfg.random_seed}'
    return os.path.join(cfg.data_name, dn_bs_dir, seed_dir)


def get_train_config(cfg):
    return {
        'editor_name': cfg.editor_name,
        'edit_model_name': cfg.edit_model_name,
        'data_name': cfg.data_name,
        'data_file_name': cfg.data_file_name,
        'split': cfg.split,
        'data_n': cfg.data_n,
        'batch_size': cfg.batch_size,
        'device': cfg.device,
        'extra_devices': cfg.extra_devices,
        'load_ckpt_path': cfg.load_ckpt_path,
        'epochs': cfg.epochs,
        'train_name_prefix': cfg.train_name_prefix,
        'save_ckpt_per_i': cfg.save_ckpt_per_i,
        'log_per_i': cfg.log_per_i,
        'ema_alpha': cfg.ema_alpha,
        'random_seed': cfg.random_seed,
        'data_buffer_size': cfg.data_buffer_size,
        'records_subdir': get_train_name(cfg),
    }

if __name__ == '__main__':
    cfg = get_attr()
    set_device(cfg)
    cfg.data_name = cfg.data_name.upper()
    resolve_random_seed(cfg)
    # load editor
    editor = load_vllm_editor(cfg, cfg.editor_name, cfg.edit_model_name, cfg.device, cfg.extra_devices, None, True)
    # load data
    if cfg.data_name == 'EVQA':
        from dataset.vllm import EVQA
        data_path = os.path.join(ROOT_PATH, 'data/easy-edit-mm/vqa/vqa_train.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/easy-edit-mm/images')
        train_data = EVQA(data_path, img_root_dir, cfg.data_n)
    elif cfg.data_name == 'EIC':
        from dataset.vllm import EIC
        data_path = os.path.join(ROOT_PATH, 'data/easy-edit-mm/caption/caption_train_edit.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/easy-edit-mm/images')
        train_data = EIC(data_path, img_root_dir, cfg.data_n)
    elif cfg.data_name == 'VLKEB':
        from dataset.vllm import VLKEB
        data_path = os.path.join(ROOT_PATH, 'data/VLKEB/train.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/VLKEB/mmkb_images')
        train_data = VLKEB(data_path, img_root_dir, cfg.data_n)
    elif cfg.data_name == 'WATERBIRD':
        from dataset.vllm import WaterBird
        data_path = os.path.join(ROOT_PATH, f'data/WaterBird/{cfg.data_file_name}.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/WaterBird')
        train_data = WaterBird(data_path, img_root_dir, cfg.split, cfg.data_n)
    elif cfg.data_name == "CUB_FILTERED":
        from dataset.vllm import WaterBird
        data_path = os.path.join(ROOT_PATH, f'data/CUB_filtered/{cfg.data_file_name}.json')
        img_root_dir = os.path.join(ROOT_PATH, 'data/CUB_filtered')
        train_data = WaterBird(data_path, img_root_dir, cfg.split, cfg.data_n)
    elif cfg.data_name == 'WATERBIRDV2':
        from dataset.vllm import WaterBirdV2
        data_path = os.path.join(ROOT_PATH, 'data/WaterBird-v2.0/edit_annotations.json')
        img_root_dir = ''
        train_data = WaterBirdV2(data_path, img_root_dir, cfg.split, cfg.data_n)
    else:
        raise ValueError(f"Unknown data name: {cfg.data_name}")
    # initialize and train
    editor.train_init(train_data, cfg.batch_size, train_name_prefix = cfg.train_name_prefix,
        train_name = get_train_name(cfg),
        load_ckpt_path = cfg.load_ckpt_path, save_ckpt_per_i = cfg.save_ckpt_per_i, 
        log_per_i = cfg.log_per_i, ema_alpha = cfg.ema_alpha, random_seed = cfg.random_seed,
        data_buffer_size = cfg.data_buffer_size, run_config = get_train_config(cfg))
    editor.train(cfg.epochs)
