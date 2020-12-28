import numpy as np
import os
import json
import pandas as pd
import time
import random
import typing
from typing import Callable, Tuple
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

# iaa
from imgaug import augmenters as iaa
import imgaug as ia
import math
ia.seed(math.floor(time.time()))

# albumentations
import albumentations
import cv2

# torch
import torch
from torchvision import transforms
from torch import Tensor

# customized library
from .util import concat_tiles, binary_search, pad_background
from hephaestus.data.openslide_wrapper_v2 import Slide_OSread
from hephaestus.data.vahadane import VahadaneNormalizer

class PathoAugmentation(object):

    ''' Customized augmentation method:
        discrete augmentation: do these augmentation to each 6*6 tiles in the image respectively
        complete augmentation: do these augmentation to the whole image at one time
    '''
    discrete_aug = albumentations.Compose([
                                           albumentations.Transpose(p=0.5),
                                           albumentations.VerticalFlip(p=0.5),
                                           albumentations.HorizontalFlip(p=0.5),
                                           albumentations.transforms.Rotate(limit=15, border_mode=cv2.BORDER_WRAP, p=0.5),
                                           albumentations.imgaug.transforms.IAAAdditiveGaussianNoise(p=0.3),
                                           albumentations.augmentations.transforms.MultiplicativeNoise (multiplier=(0.95, 1.05), elementwise=True, p=0.5),
                                           albumentations.augmentations.transforms.HueSaturationValue(hue_shift_limit=10, 
                                                                                                      sat_shift_limit=10, 
                                                                                                      val_shift_limit=10, p=0.3)
                                           ])
    complete_aug = albumentations.Compose([])


def get_resnet_preproc_fn():
    return  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])

def get_n_img(dir: str, num:int = 1):
    files = [file for file in os.listdir(dir)]
    random.shuffle(files)
    return [imread(os.path.join(dir, file)) for file in files[:num]]
    
class VahadaneWrapper:
    def __init__(self, csv1:str, csv2:str, img_dir:str, vahadane1:str=None, vahadane2:str=None):
        self.img_dir = img_dir
        
        CSV1 = pd.read_csv(csv1)
        CSV2 = pd.read_csv(csv2)
        self.dict1 = {num:None for num in CSV1['patho_number']}
        self.dict2 = {num:None for num in CSV2['patho_number']}
        
        if vahadane1 is not None and os.path.isfile(vahadane1):
            print(f"Loading two vahadane normalizer")
            self.vahadane1 = VahadaneNormalizer()
            self.vahadane1.load_stain_matrix(vahadane1)
            self.vahadane2 = VahadaneNormalizer()
            self.vahadane2.load_stain_matrix(vahadane2)
        else:
            self.vahadane1, self.vahadane2 = self._fit_domain()
    
    def _fit_domain(self) -> Tuple[VahadaneNormalizer, VahadaneNormalizer]:
        first_domain_images  = []
        second_domain_images = []
        for dir_name in os.listdir(self.img_dir):
            short_name = (dir_name[:13])
            if short_name in dict1:
                first_domain_images.extend(get_n_img(os.path.join(self.img_dir, dir_name), 10))
            elif short_name in dict2:
                second_domain_images.extend(get_n_img(os.path.join(self.img_dir, dir_name), 10))
            else:
                print(f'File {dir_name} is not found in the CSV')
        vahadane1 = VahadaneNormalizer()
        vahadane1.fit(images=first_domain_images)
        vahadane2 = VahadaneNormalizer()
        vahadane2.fit(images=second_domain_images)
        #vahadane1.save_stain_matrix('./checkpoint/vahadane1')
        #vahadane2.save_stain_matrix('./checkpoint/vahadane2')
        return vahadane1, vahadane2
        
    def _check_domain(self, short_name:str)-> int:
        if short_name in self.dict1:
            return 0
        elif short_name in self.dict2:
            return 1
        else:
            '''image that doesn't belong to domain 1 or 2, will not be normalized'''
            return 0
    
    def fit(self, img_list: list, name:str) -> list:
        if self._check_domain(name) == 1:
            img_list = [self.vahadane1.normalize(img_list[i],image_normalizer=self.vahadane2) 
                        for i in range(len(img_list))]
        return img_list
        
# Dataloader for concated tiles
class TileDataset:
    def __init__(self, 
                 img_dir: str, 
                 json_dir: str, 
                 patch_size: int, 
                 patch_dir: str = None,
                 resize_ratio: int =1,
                 tile_size: int=6, 
                 dup: int=4, 
                 is_test: bool =False, 
                 aug: object=None, 
                 preproc: Callable=None,
                 debug: bool =False,
                 balance:bool = False,
                 stain_wrapper: VahadaneWrapper = None,
                ) -> None:
        '''
        Arg: img_names: string, the slide names derived from train_test split
             img_dir: path string, the directory of the slide tiff files
             json_dir: path string, the directory of the json files
             patch_size: int
             tile_size: int, each item from this dataset will be rendered as a concated 3*tile_size*tile_size image
             dup: int, will get $dup concatenated tiles from one slide to save slide reading time
             aug: augmentation function
             preproc: torch transform object, default: resnet_preproc
        '''
        self.json_dir     = json_dir
        self.img_dir      = img_dir
        self.patch_size   = patch_size
        self.patch_dir    = patch_dir
        self.resize_ratio = resize_ratio
        self.tile_size    = tile_size
        self.dup          = dup
        self.is_test      = is_test
        self.aug          = aug
        self.preproc      = preproc
        self.debug        = debug
        self.stain_wrapper=stain_wrapper
        
        # the following 3 list should be kept in the same order!
        self.img_names   = [img_name.split('.ndpi')[0] for img_name in os.listdir(img_dir)]
        self.her2_titer  = []
        self.coords      = None
        
        self._get_her2_titer()
        self._get_coord()
        
        if balance:
            self._balance_pos_neg()

        self.cur_pos = 0

    def _get_img_names(self) ->list:
        if self.patch_dir:
            return os.listdir(self.patch_dir)
        else:
            return [img_name.split('.ndpi')[0] for img_name in os.listdir(img_dir)]

    def _balance_pos_neg(self)->None:
        for i in range(len(self.img_names)):
            if self.her2_titer[i] != 3:
                self.img_names.append(self.img_names[i])
                self.her2_titer.append(self.her2_titer[i])
                if not self.patch_dir:
                    self.coords.append(self.coords[i])

    def _jugde_titer_by_name(self, name: str) -> int:
        if 'FISH' in name and name[-1] == '1':
            titer=4
        else:
            titer = int(name[-1])
        return titer
    
    def _get_her2_titer(self) -> None:
        '''
            FISH positive ==> titer 4
            otherwise titer <- IHC
        '''
        self.her2_titer.clear()
        for name in self.img_names:
            self.her2_titer.append(self._jugde_titer_by_name(name))
    
    def _get_coord(self) -> None:
        # get the coordinated prefetched by /foreground/preprocess.py, stored at /foreground/data_256/*.json
        if self.patch_dir:
            return
        self.coords = []
        for img_name in self.img_names:
            with open(os.path.join(self.json_dir, img_name+'.json')) as f:
                js = json.load(f)
            tmp_list = (js[img_name][0]['coord_list'])
            self.coords.append(tmp_list)
        
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        '''
        This function will fectch idx-th bags of image from the coordinate list, get the first
        tile size * tile size images from the coordinate list, concatenate them, and performing augmentation, 
        preprocessing. In addition, the label will also be generated as a torch tensor
        
            grade 3 --> [1, 1, 1, 0, 0]
            grade 5 --> [1, 1, 1, 1, 1]
            grade 0 --> [0, 0, 0, 0, 0]
        
        Ultimately, the preprocessed image will be returned altogether with the groundtruth label
        
        Return:
            Training time:
                imgs: a tensor
                label: a tensor
            Testing time:
                imgs: list of tensor with length self.dup --> will be averaged after passing to model
                label: ONE tensor
        '''
        self.cur_pos = idx

        # open slide by Slide_OSread or skimage.io.MultiImage
        this_slide = self._read_slide(idx) if self.patch_dir is None else None

        # cat tile is of type "np.float32", had been normalized to 1
        # test time augmentation: find 4 images and will average their predicted values
        if not self.is_test:
            img = torch.from_numpy(self._get_one_cat_tile(idx, this_slide).transpose((2, 0, 1))).float()
            if self.preproc is not None:
                img = self.preproc(img)
        else:
            cat_tiles = [self._get_one_cat_tile(idx, this_slide) for i in range(self.dup)]
            img = [torch.from_numpy(np.transpose(cat_tile, (2, 0, 1))).float() for cat_tile in cat_tiles]
            if self.preproc is not None:
                img = [self.preproc(im) for im in img]
            img = torch.stack(img)

        # get label
        label = np.zeros(4)
        for i in range(0, self.her2_titer[idx]):
            label[i] = 1
        label = torch.from_numpy(label).float()
        
        self.cur_pos = (self.cur_pos+1)%len(self)
        return img, label
    
    def _read_slide(self, idx: int) -> Slide_OSread:
        '''Read slide by openslide reader if no resize, skimage.io.multiimage if resize'''
        return Slide_OSread(os.path.join(self.img_dir, self.img_names[idx]+".ndpi"), show_info=False)
    
    def _trim_name_to_shortname(self, name:str)->str:
        '''take first 13 character from the string'''
        return name[:13]
    
    def _get_one_cat_tile(self, idx:int, this_slide: Slide_OSread) -> np.ndarray:
        '''
        Argument: 
            this_slide: 
                1. openslide object(slower) OR
                2. skimage(faster)
            idx: int, position of the idx-th tuple in coordinate list
        Return: cat_tile: an numpy array sized 1536*1536*3 with type float32, normalized to 0-1

        No matter the value of patch size, the acquired image patches will be resized to 256*256 and
        be concatenated to a 1536*1536*3 image
        If the patch size is 256, or 512, then get patch from openslide object
        If the patch size is 1024, then get patch from the first image pyramid of skimage
        '''
        ps, ts = self.patch_size, self.tile_size
        dst_sz = self.patch_size //self.resize_ratio
            
        if not self.patch_dir:
            chosen_indexs = np.random.choice(len(self.coords[idx]), min(ts**2, len(self.coords[idx])), replace=False)
            foreground_coord = [self.coords[idx][chosen_index] for chosen_index in chosen_indexs]

            tiles = [this_slide.get_patch_at_level(coord=(x*ps, y*ps),
                                                   sz=(dst_sz, dst_sz),
                                                   level=int(math.log2(self.resize_ratio)),
                                                  ) for x, y in foreground_coord]
        else:
            target_dir    = os.path.join(self.patch_dir, self.img_names[idx])
            target_len    = len(os.listdir(target_dir))
            chosen_indexs = np.random.choice(target_len, min(ts**2, target_len), replace=False)
            all_tiles     = os.listdir(target_dir)
            tiles = [imread(os.path.join(target_dir, all_tiles[index])) 
                     for index in chosen_indexs]
        
        # stain normalization
        if self.stain_wrapper:
            tiles = self.stain_wrapper.fit(tiles, self._trim_name_to_shortname(self.img_names[idx]))
        
        # remove background
        tiles = [pad_background(tile) for tile in tiles]

        # discrete augmentation will be performed image-wise before concatenation
        # complete augmentation will be performed to the concatenated big image 
        if self.aug is not None:
            tiles = [self.aug.discrete_aug(image=tile)['image'] for tile in tiles]
            cat_tile = self.aug.complete_aug(image=concat_tiles(tiles, patch_size=dst_sz, tile_sz=self.tile_size))['image']
            cat_tile = cat_tile
        else:
            cat_tile = concat_tiles(tiles, patch_size=dst_sz, tile_sz=self.tile_size)        
        return cat_tile.astype(np.float32)/255.
    
    def __next__(self):
        return self.__getitem__(self.cur_pos)
    
    def __len__(self) -> int:
        if self.debug:
            return 10
        return len(self.img_names)