import os


# Resolve paths from this repository so the code remains portable after migration.
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
model_path_map = {
    'blip2-opt-2.7b': 'Salesforce/blip2-opt-2.7b',
    'llava-v1.5-7b': 'llava-hf/llava-1.5-7b-hf',
    'minigpt-4-vicuna-7b': os.path.join(ROOT_PATH, 'minigpt-4-vicuna-7b'),
    'qwen2-vl-7b': 'Qwen/Qwen2-VL-7B-Instruct',
}
