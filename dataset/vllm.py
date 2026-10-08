#%%
from typing import Dict, List, Tuple, Union
from . import BaseEditData
from copy import deepcopy
import torch, os, json, re
from PIL import Image
from tqdm import tqdm
import pandas as pd


class BaseVLLMEditData(BaseEditData):
    '''
    Functions used to read and preprocess VLLM editing datasets, which should be
        structured as a list like [
            { # test1
                'requests': [
                    {'image': PILImage, 'prompt': str, 'target_new': str, ...},
                    {'image': PILImage, 'prompt': str, 'target_new': str, ...}, ...
                ],
                'generality': {
                    'gen_1_name':[
                        {'image': PILImage, 'prompt': str, 'target': str, ...},
                        {'image': PILImage, 'prompt': str, 'target': str, ...}, ...
                    ],
                    'gen_2_name':[...], ...
                },
                'locality': {
                    'loc_1_name':[
                        {'image': PILImage, 'prompt': str, 'target': str, ...}, ...
                    ],
                    'loc_2_name':[...], ...
                }
            }, 
            { # test2
                'requests': ...
            }, ...
        ]
    '''
    def __init__(self, data_with_img, data_with_img_path) -> None:
        super().__init__(data_with_img) 
        self.data = data_with_img
        self.data_with_img = data_with_img
        self.data_with_img_path = data_with_img_path

    def __load_imgs_for_data_with_img_path__(self, d:Union[List, Dict, str]):
        if isinstance(d, dict):
            for k in list(d.keys()):
                if k == 'image':
                    if d[k] != None:
                        d[k + '_path'] = d[k]
                        d[k] = Image.open(d[k])
                else:
                    self.__load_imgs_for_data_with_img_path__(d[k])
        elif isinstance(d, list):
            for i in d:
                self.__load_imgs_for_data_with_img_path__(i)
        elif isinstance(d, str): return
        else: raise
    
    def get_data_with_img_path(self):
        return self.data_with_img_path

    def __init_eic_evqa__(self, data_path:str, img_root_dir:str, data_n = None):
        if data_n == None: data_n = 99999999
        with open(data_path, 'r') as f:
            data = json.load(f)
        data_n = min(len(data), data_n)
        return_data = []
        for i in tqdm(range(data_n), 'Loading data'):
            d = data[i]
            new_d = {'requests': [{}], 
                     'generality': {'text_rephrase': [], 'image_rephrase': []}, 
                     'locality': {'text_loc': [], 'image_loc': []}}
            # requests
            new_d['requests'][0]['image'] = os.path.join(img_root_dir, d['image'])
            new_d['requests'][0]['prompt'] = d['src']
            new_d['requests'][0]['target_new'] = d['alt']
            # generality
            a_gen_data = {'image': new_d['requests'][0]['image'], 'prompt': d['rephrase'], 'target': d['alt']}
            new_d['generality']['text_rephrase'].append(a_gen_data)
            a_gen_data = {'image': os.path.join(img_root_dir, d['image_rephrase']), 
                          'prompt': d['src'], 'target': d['alt']}
            new_d['generality']['image_rephrase'].append(a_gen_data)
            # locality
            a_loc_data = {'image': None, 'prompt': d['loc'], 'target': d['loc_ans']}
            new_d['locality']['text_loc'].append(a_loc_data)
            a_loc_data = {'image': os.path.join(img_root_dir, d['m_loc']), 
                          'prompt': d['m_loc_q'], 'target': d['m_loc_a']}
            new_d['locality']['image_loc'].append(a_loc_data)
            return_data.append(new_d)
        return return_data


class EVQA(BaseVLLMEditData):
    def __init__(self, data_path:str = 'data/easy-edit-mm/vqa/vqa_train.json', 
                  img_root_dir:str = 'data/easy-edit-mm/images', data_n = None) -> None:
        if 'vqa' not in os.path.basename(data_path): raise
        print('Load E-VQA from: %s '% data_path)
        data_with_img_path = self.__init_eic_evqa__(data_path, img_root_dir, data_n)
        for d in data_with_img_path:
            d['requests'][0]['prompt'] = '%s The answer is:'%d['requests'][0]['prompt']
            d['generality']['text_rephrase'][0]['prompt'] = '%s The answer is:'%d['generality']['text_rephrase'][0]['prompt']
            d['generality']['image_rephrase'][0]['prompt'] = '%s The answer is:'%d['generality']['image_rephrase'][0]['prompt']
            d['locality']['text_loc'][0]['prompt'] = '%s?'%d['locality']['text_loc'][0]['prompt']
            d['locality']['image_loc'][0]['prompt'] = '%s The answer is:'%d['locality']['image_loc'][0]['prompt']
        data_with_img = deepcopy(data_with_img_path)
        for d in tqdm(data_with_img, 'Loading images'):
            self.__load_imgs_for_data_with_img_path__(d)
        super().__init__(data_with_img, data_with_img_path)

    def dataset_name(self):
        return 'EVQA'


class EIC(BaseVLLMEditData):
    def __init__(self, data_path:str = 'data/easy-edit-mm/caption/caption_train_edit.json', 
                  img_root_dir:str = 'data/easy-edit-mm/images', data_n = None):
        if 'caption' not in os.path.basename(data_path): raise
        print('Load E-IC from: %s '% data_path)
        data_with_img_path = self.__init_eic_evqa__(data_path, img_root_dir, data_n)
        for d in data_with_img_path:
            d['locality']['text_loc'][0]['prompt'] = '%s?'%d['locality']['text_loc'][0]['prompt']
            d['locality']['image_loc'][0]['prompt'] = '%s The answer is:'%d['locality']['image_loc'][0]['prompt']
        data_with_img = deepcopy(data_with_img_path)
        for d in tqdm(data_with_img, 'Loading images'):
            self.__load_imgs_for_data_with_img_path__(d)
        super().__init__(data_with_img, data_with_img_path)

    def dataset_name(self):
        return 'EIC'


class VLKEB(BaseVLLMEditData):
    def __init__(self, data_path:str = 'data/VLKEB/train.json', 
                  img_root_dir:str = 'data/VLKEB/mmkb_images', data_n = None):
        print('Load VLKEB from: %s '% data_path)
        data_with_img_path = self.__init_eic_evqa__(data_path, img_root_dir, data_n)
        for d in data_with_img_path:
            d['locality']['text_loc'][0]['prompt'] = '%s?'%d['locality']['text_loc'][0]['prompt']
            d['locality']['image_loc'][0]['prompt'] = '%s The answer is:'%d['locality']['image_loc'][0]['prompt']
        data_with_img = deepcopy(data_with_img_path)
        for d in tqdm(data_with_img, 'Loading images'):
            self.__load_imgs_for_data_with_img_path__(d)
        super().__init__(data_with_img, data_with_img_path)

    def dataset_name(self):
        return 'VLKEB'


class SpuMNIST(BaseVLLMEditData):
    def __init__(self, data_path:str = 'data/SpuMNIST/train.json', img_root_dir:str = 'data/SpuMNIST/images', data_n = None):
        print('Load SpuMNIST from: %s '% data_path)
        data_with_img_path = self.__init_eic_evqa__(data_path, img_root_dir, data_n)
        for d in data_with_img_path:
            d['requests'][0]['prompt'] = '%s The answer is:'%d['requests'][0]['prompt']
            d['generality']['text_rephrase'][0]['prompt'] = '%s The answer is:'%d['generality']['text_rephrase'][0]['prompt']
            d['generality']['image_rephrase'][0]['prompt'] = '%s The answer is:'%d['generality']['image_rephrase'][0]['prompt']
            d['locality']['text_loc'][0]['prompt'] = '%s?'%d['locality']['text_loc'][0]['prompt']
            d['locality']['image_loc'][0]['prompt'] = '%s The answer is:'%d['locality']['image_loc'][0]['prompt']
        data_with_img = deepcopy(data_with_img_path)
        for d in tqdm(data_with_img, 'Loading images'):
            self.__load_imgs_for_data_with_img_path__(d)
        super().__init__(data_with_img, data_with_img_path)

    def dataset_name(self):
        return 'SpuMNIST'
    
    def __init_eic_evqa__(self, data_path:str, img_root_dir:str, data_n = None):
        if data_n == None: data_n = 99999999
        with open(data_path, 'r') as f:
            data = json.load(f)
        metadata = data['metadata']
        data = data['annotations']
        data_n = min(len(data), data_n)
        return_data = []
        for i in tqdm(range(data_n), 'Loading data'):
            d = data[i]
            new_d = {'requests': [{}], 
                     'generality': {'text_rephrase': [], 'image_rephrase': []}, 
                     'locality': {'text_loc': [], 'image_loc': []}}
            # requests
            new_d['requests'][0]['image'] = os.path.join(img_root_dir, d['image'])
            new_d['requests'][0]['prompt'] = d['src']
            new_d['requests'][0]['target_new'] = d['alt']
            # generality
            a_gen_data = {'image': new_d['requests'][0]['image'], 'prompt': d['rephrase'], 'target': d['alt']}
            new_d['generality']['text_rephrase'].append(a_gen_data)
            a_gen_data = {'image': os.path.join(img_root_dir, d['rephrase_image']), 
                          'prompt': d['src'], 'target': d['alt']}
            new_d['generality']['image_rephrase'].append(a_gen_data)
            # locality
            a_loc_data = {'image': None, 'prompt': d['loc'], 'target': d['loc_ans']}
            new_d['locality']['text_loc'].append(a_loc_data)
            a_loc_data = {'image': os.path.join(img_root_dir, d['m_loc']), 
                          'prompt': d['m_loc_q'], 'target': d['m_loc_a']}
            new_d['locality']['image_loc'].append(a_loc_data)
            return_data.append(new_d)
        return return_data
    

class WaterBird(BaseVLLMEditData):
    def __init__(self, data_path:str = 'data/WaterBird/edit_annotations.json', 
                  img_root_dir:str = 'data/WaterBird', split:str = 'train', data_n = None):
        print('Load WaterBird from: %s '% data_path)
        data_with_img_path = self.__init_eic_evqa__(data_path, img_root_dir, split, data_n)
        for d in data_with_img_path:
            d['requests'][0]['prompt'] = '%s The answer is:'%d['requests'][0]['prompt']
            d['generality']['text_rephrase'][0]['prompt'] = '%s The answer is:'%d['generality']['text_rephrase'][0]['prompt']
            d['generality']['image_rephrase'][0]['prompt'] = '%s The answer is:'%d['generality']['image_rephrase'][0]['prompt']
            d['locality']['text_loc'][0]['prompt'] = '%s?'%d['locality']['text_loc'][0]['prompt']
            d['locality']['image_loc'][0]['prompt'] = '%s The answer is:'%d['locality']['image_loc'][0]['prompt']
        data_with_img = deepcopy(data_with_img_path)
        for d in tqdm(data_with_img, 'Loading images'):
            self.__load_imgs_for_data_with_img_path__(d)
        super().__init__(data_with_img, data_with_img_path)

    def dataset_name(self):
        return 'WaterBird'
    
    def _get_split(self, data, split):
        assert split in ['train', 'val', 'test', 'all']
        return data if split == 'all' else [d for d in data if d['split'] == split]
    
    def __init_eic_evqa__(self, data_path:str, img_root_dir:str, split:str, data_n = None):
        if data_n == None: data_n = 99999999
        with open(data_path, 'r') as f:
            data = json.load(f)
        data = self._get_split(data, split)
        data_n = min(len(data), data_n)
        return_data = []
        for i in tqdm(range(data_n), 'Loading data'):
            d = data[i]
            new_d = {'requests': [{}], 
                     'generality': {'text_rephrase': [], 'image_rephrase': []}, 
                     'locality': {'text_loc': [], 'image_loc': []}}
            # requests
            new_d['requests'][0]['image'] = os.path.join(img_root_dir, d['image'])
            new_d['requests'][0]['prompt'] = d['src']
            new_d['requests'][0]['target_new'] = d['alt']
            new_d['requests'][0]['desc'] = f"{d['water']} bird on {d['place']}"
            # generality
            a_gen_data = {'image': new_d['requests'][0]['image'], 'prompt': d['rephrase'], 'target': d['alt']}
            new_d['generality']['text_rephrase'].append(a_gen_data)
            a_gen_data = {'image': os.path.join(img_root_dir, d['image_rephrase']), 
                          'prompt': d['src'], 'target': d['alt']}
            new_d['generality']['image_rephrase'].append(a_gen_data)
            # locality
            a_loc_data = {'image': None, 'prompt': d['loc_q'], 'target': d['loc_a']}
            new_d['locality']['text_loc'].append(a_loc_data)
            a_loc_data = {'image': os.path.join("data/easy-edit-mm/images", d['m_loc_img']), 
                          'prompt': d['m_loc_q'], 'target': d['m_loc_a']}
            new_d['locality']['image_loc'].append(a_loc_data)
            return_data.append(new_d)
        return return_data


class WaterBirdV2(BaseVLLMEditData):
    """WaterBird-v2.0: toy benchmark with real CUB images,
    image-rephrase as spurious correlation probe. Paths are project-root-relative."""

    def __init__(self, data_path: str = 'data/WaterBird-v2.0/edit_annotations.json',
                 img_root_dir: str = '', split: str = 'train', data_n = None):
        if not os.path.isabs(data_path):
            repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(repository_root, data_path)
        data_path = os.path.normpath(data_path)
        if not img_root_dir:
            img_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(data_path)))
        print('Load WaterBird-v2.0 from: %s' % data_path)
        data_with_img_path = self.__init_eic_evqa__(data_path, img_root_dir, split, data_n)
        for d in data_with_img_path:
            d['requests'][0]['prompt'] = '%s The answer is:' % d['requests'][0]['prompt']
            d['generality']['text_rephrase'][0]['prompt'] = '%s The answer is:' % d['generality']['text_rephrase'][0]['prompt']
            d['generality']['image_rephrase'][0]['prompt'] = '%s The answer is:' % d['generality']['image_rephrase'][0]['prompt']
            d['locality']['text_loc'][0]['prompt'] = '%s?' % d['locality']['text_loc'][0]['prompt']
            d['locality']['image_loc'][0]['prompt'] = '%s The answer is:' % d['locality']['image_loc'][0]['prompt']
            # Store spurious-correlation metadata
            d['requests'][0]['desc'] = f"{d.get('water','?')} bird on {d.get('place','?')}"
            d['generality']['image_rephrase'][0]['desc'] = f"rephrase: {d.get('image_rephrase_place','?')} bg"
        data_with_img = deepcopy(data_with_img_path)
        for d in tqdm(data_with_img, 'Loading images'):
            self.__load_imgs_for_data_with_img_path__(d)
        super().__init__(data_with_img, data_with_img_path)

    def dataset_name(self):
        return 'WaterBirdv2'

    def _get_split(self, data, split):
        assert split in ['train', 'test', 'all']
        return data if split == 'all' else [d for d in data if d['split'] == split]

    def __init_eic_evqa__(self, data_path: str, img_root_dir: str, split: str, data_n=None):
        if data_n is None: data_n = 99999999
        with open(data_path, 'r') as f:
            data = json.load(f)
        data = self._get_split(data, split)
        data_n = min(len(data), data_n)
        return_data = []
        for i in tqdm(range(data_n), 'Loading data'):
            d = data[i]
            new_d = {'requests': [{}],
                     'generality': {'text_rephrase': [], 'image_rephrase': []},
                     'locality': {'text_loc': [], 'image_loc': []}}
            # requests
            new_d['requests'][0]['image'] = os.path.join(img_root_dir, d['image'])
            new_d['requests'][0]['prompt'] = d['src']
            new_d['requests'][0]['target_new'] = d['alt']
            # generality: text_rephrase
            a_gen_data = {'image': new_d['requests'][0]['image'], 'prompt': d['rephrase'], 'target': d['alt']}
            new_d['generality']['text_rephrase'].append(a_gen_data)
            # generality: image_rephrase (spurious probe)
            a_gen_data = {'image': os.path.join(img_root_dir, d['image_rephrase']),
                          'prompt': d['src'], 'target': d['alt']}
            new_d['generality']['image_rephrase'].append(a_gen_data)
            # locality: text
            a_loc_data = {'image': None, 'prompt': d['loc_q'], 'target': d['loc_a']}
            new_d['locality']['text_loc'].append(a_loc_data)
            # locality: image
            a_loc_data = {'image': os.path.join(img_root_dir, d['m_loc_img']),
                          'prompt': d['m_loc_q'], 'target': d['m_loc_a']}
            new_d['locality']['image_loc'].append(a_loc_data)
            return_data.append(new_d)
        return return_data


class CUBInferenceDataset:
    def __init__(self, data_path:str="data/CUB_200_2011/CUB_200_2011/metadata.csv", 
                 img_dir:str="data/CUB_200_2011/CUB_200_2011/images", size:int=None):
        self.data_path = data_path
        self.image_dir = img_dir
        
        instruction = "Please answer with either water or land, and do not include any additional punctuation, numbers, or characters."
        instruction_given_spu = "The background of the image may spuriously correlated to the prediction, but it should not be considered. Please focus on the bird itself and answer with water or land."

        text = [
            # "Is the bird in the image a water bird? Please only answer yes or no.",
            # "Is the bird in the image a land bird? Please only answer yes or no.",
            "What type of bird is in the image? Choose from: land, water.",
            f"Does the bird in the picture belong to a water bird or a land bird? {instruction}",
            "Look at this bird image. Is this a water bird or a land bird? Please answer with 'water' or 'land'.",
            f"Identify whether the bird shown is a water bird or a land bird. {instruction_given_spu}",
            "This image shows a bird in a {} background. Please identify if this bird is a water bird or a land bird.",
            # "What species of bird is shown? Answer me with reasons.",
        ]
        
        # Load annotations
        anns = pd.read_csv(self.data_path)
        data = []
        for i, row in tqdm(anns.iterrows(), total=len(anns), desc=f'Loading CUB-200-2011 from {self.data_path}'):
            img_id = row['image_id']
            img_path = os.path.join(self.image_dir, row['image_path'])
            img = Image.open(img_path).convert('RGB')
            bird_type = row['y']
            background = row['background'] if 'background' in row else None
            data.append({'img_id': img_id, 'img_path': img_path, 'image': img, 'bird_type': bird_type, 'background': background, 'text': text})
            
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# d = VLKEB(data_n=10)
# #%%
# d.data[0]['requests'][0]['image'].show()
# d.data[0]['requests']
# d.data[0]['generality']
# d.data[0]['locality']
