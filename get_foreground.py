import os
import numpy as np
import math
import pandas as pd # reading label csv
from tqdm import tqdm # only used in jupyter notebook
from skimage import color
import json # save foreground info into a json file
from scipy.ndimage.morphology import binary_dilation
from scipy import ndimage
from joblib import Parallel, delayed
import skimage.io

from hephaestus.data.openslide_wrapper_v2 import Slide_OSread

SLIDE_DIR = './train_images/'
MIL_PATCH_SIZE = 2048
MIL_RESIZE_RATIO = 8
FOREGROUND_DIR = f'./train_bbox/data_{MIL_PATCH_SIZE}/'
EXTENSION = '.ndpi'

def acquire_image(path, level):
    try:
        slide = Slide_OSread(path, show_info=False)
    except:
        print(f'slide {path} cannot be found')
    return slide.get_patch_at_level((0, 0), slide.slide.level_dimensions[level], level=level)

def compute_mean_saturation(image_patch):
    image_mean = np.mean(image_patch, axis=(0, 1)).reshape(1, 1, 3)
    return color.rgb2hsv(image_mean)[0, 0, 1]

def find_foreground_patches(path, patch_size, threshold=0.05, resize_ratio=4):
    '''
    Arg:
        path: image path string
        is_mask: if the path is mask path or not
        orig_path: string, if the path is mask path, user should render the original image path, using
                   the original image for background filtering
        patch_size: int, normally 256
        threshold: float, threshold for hsv saturation filtering
    Return: a list of (x, y) left upper corner of patches that contains > 20% non-background pixels
        This function will first apply scipy mask expansion to a resized foreground mask and judge if these
    patches belong to foregound using the saturation of the mean value of the patch.
    '''
    this_slide = acquire_image(path, int(math.log2(resize_ratio)))
    
    H, W = this_slide.shape[:2]
    H *= resize_ratio
    W *= resize_ratio
    w = W//patch_size
    h = H//patch_size

    obj_coord_list = []
    ps = patch_size//resize_ratio
    for hh in range(h):
        for ww in range(w):
            image_patch = this_slide[hh*ps:(hh+1)*ps, ww*ps:(ww+1)*ps]
            if compute_mean_saturation(image_patch) >= threshold:
                obj_coord_list.append((ww, hh))
    return obj_coord_list


def save_foreground_info(save_path, slide_name, coord_list, from_mask=0):
    if os.path.isfile(save_path):
        print(f"File {save_path} exits, overwritting...")
    mil_dict={}
    mil_dict[slide_name] = []
    mil_dict[slide_name].append({"coord_list": coord_list, "from_mask": from_mask})
    with open(save_path, 'w') as fd:
        json.dump(mil_dict, fd)

def process_file_list(target_file, from_mask):
    if os.path.isfile(os.path.join(FOREGROUND_DIR, target_file+".json")):
        print(f"File {target_file} had been worked before, skipping...")
        return
    c = find_foreground_patches(os.path.join(SLIDE_DIR, target_file+EXTENSION), 
                                patch_size = MIL_PATCH_SIZE)
    save_foreground_info(os.path.join(FOREGROUND_DIR, target_file+".json"), slide_name=target_file, coord_list=c, from_mask=from_mask)
    
if __name__ == "__main__":
    img_files  = [f.split(EXTENSION)[0] for f in os.listdir(SLIDE_DIR)]
    
    print("Check foreground directory exist?")
    if not os.path.isdir(FOREGROUND_DIR):
        os.makedirs(FOREGROUND_DIR)
    
    print("Processing normal files...")
    Parallel(n_jobs=10)(delayed(process_file_list) (img_file, 0) for img_file in tqdm(img_files))
    
    
