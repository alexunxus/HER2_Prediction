import os
import numpy as np
import math
from skimage.io import imsave
from skimage.exposure import is_low_contrast
from tqdm import tqdm
from joblib import Parallel, delayed
import typing
from typing import Tuple, Callable, Union

from mil_model.dataloader import TileDataset
from mil_model.config import get_cfg_defaults
from hephaestus.data.openslide_wrapper_v2 import Slide_OSread
                                                                                   
from torch import Tensor

def get_imgs(dataset: object, index:int, extension:str='.ndpi') -> list:
    img_dir    = dataset.img_dir
    slide_name = dataset.img_names[index]
    img_path = os.path.join(img_dir, slide_name+extension)
    if not os.path.isfile(img_path):
        raise ValueError(f'Path not found')
    this_slide   = Slide_OSread(img_path, show_info=False)
    ps           = dataset.patch_size
    dst_sz       = dataset.patch_size//dataset.resize_ratio
    resize_ratio = dataset.resize_ratio
    foreground_coord = dataset.coords[index]
    tiles = [this_slide.get_patch_at_level(coord=(x*ps, y*ps),
                                           sz=(dst_sz, dst_sz),
                                           level=int(math.log2(resize_ratio)),
                                          ) for x, y in foreground_coord]
    tiles = [tile for tile in tiles if not is_low_contrast(tile)]
    return tiles

def objectiveness(img: np.ndarray, threshold: int=0.1, show_img: bool=False) -> np.ndarray:
    reduced_image = rgb2hsv(pyramid_reduce(img, downscale=16,  multichannel=True))[...,1]
    if show_img:
        compare_two_image(img, reduced_image)
    return np.sum(reduced_image > threshold)*1./reduced_image.size

def save_img(img: [Tensor, np.ndarray], path:str, threshold: float = 0.4) -> None:
    if isinstance(img, Tensor):
        img= img.numpy()
    if objectiveness(img) > threshold:
        imsave(path, img)


def main():
    cfg = get_cfg_defaults()

    targets = {
        'train': [cfg.DATASET.TRAIN_IMG_DIR, cfg.DATASET.TRAIN_BBOX, 'train_patches'],
        'valid': [cfg.DATASET.VALID_IMG_DIR, cfg.DATASET.VALID_BBOX, 'valid_patches'],
        'test':  [cfg.DATASET.TEST_IMG_DIR , cfg.DATASET.TEST_BBOX , 'test_patches' ],
    }

    target_key = 'train'
    train_dataset = TileDataset(img_dir   =targets[target_key][0], 
                                json_dir  =targets[target_key][1], 
                                patch_size=cfg.DATASET.PATCH_SIZE, 
                                resize_ratio=cfg.DATASET.RESIZE_RATIO,
                                tile_size =cfg.DATASET.TILE_SIZE, 
                                is_test   =False,
                                aug       =None, 
                                preproc   =None)

    for i in tqdm(range(len(train_dataset))):
        tiles = get_imgs(train_dataset, i, '.ndpi')
        dir = f'./{targets[target_key][2]}/{train_dataset.img_names[i]}'
        os.makedirs(dir,exist_ok=True)
        r = Parallel(n_jobs=10)(delayed(save_img)(tile, os.path.join(dir, str(j).zfill(4)+'.tiff')) for j, tile in enumerate(tiles))

if __name__=='__main__':
    main()