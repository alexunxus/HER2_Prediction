import numpy as np
import random # shuffling
import pandas as pd
import os
from skimage.color import rgb2hsv
import skimage
import typing
from typing import Tuple, Callable

from torch import nn # replace bn

def replace_bn(model:nn.Module, new_norm: Callable) -> nn.Module:
    '''replace bn of model to gn'''
    for name, module in model.named_children():
        if len(list(module.named_children())):
            model._modules[name] = replace_bn(module, new_norm)
        elif type(module) == nn.BatchNorm2d:# or type(module) == nn.BatchNorm1d:
            layer_new = new_norm(module.num_features)
            #del model._modules[name]
            model._modules[name]=layer_new
    return model

def concat_tiles(tiles, patch_size:int, tile_sz:int=6) -> np.ndarray:
    '''concat tiles into (3, tile_sz*patch_size, tile_size*patch_size) image'''
    ret = np.ones((patch_size*tile_sz, patch_size*tile_sz, 3), dtype=np.uint8)*255
    if len(tiles) == 0:
        return ret
    for i in range(tile_sz**2):
        tile = tiles[i%len(tiles)]
        h, w = i//tile_sz, i%tile_sz
        ret[h*patch_size:h*patch_size + tile.shape[0], w*patch_size: w*patch_size+tile.shape[1]] = tile
    return ret

def shuffle_two_arrays(imgs: list, labels: list)->Tuple[list, list]:
    zip_obj = list(zip(imgs, labels))
    random.shuffle(zip_obj)
    return zip(*zip_obj)

def binary_search(li, x, lo: int, hi:int):
    mid = (lo+hi)//2
    if li[mid][0] == x:
        return li[mid][1]
    if mid == hi:
        raise ValueError(f"{x} is not in list!")
    if li[mid][0] > x:
        return binary_search(li, x, lo, mid)
    else:
        return binary_search(li, x, mid+1, hi)

def pad_background(img: np.ndarray, saturation_threshold: float = 0.1, 
                   resize_ratio :int =64)-> np.ndarray:
    '''
    Padding background to 255
    
    :Usage
        img = (np.rand(32, 32, 3)*255).astype(np.uint8)
        out = pad_background(img)
    
    :Parameters:
        channel last np.ndarray, dtype uint8, range 0~255
        saturation_threshold: cut-off for background filtering, default 0.1
    '''
    assert img.shape[0] == img.shape[1], (f'image shape mismatch! ({img.shape}), dim0 != dim1')
    resized_image = skimage.measure.block_reduce(img, (resize_ratio,resize_ratio, 1), np.mean)
    hsv_image = rgb2hsv(resized_image)
    foreground_mask = (hsv_image[...,1] > saturation_threshold)
    foreground_mask = skimage.transform.resize(foreground_mask, 
                                               output_shape=(i*resize_ratio for i in foreground_mask.shape))
    foreground_mask = np.repeat(np.expand_dims(foreground_mask, axis=-1), img.shape[2], axis=-1).astype(np.uint8)
    out_img = 255-(255-img)*foreground_mask
    return out_img

class Metric:
    def __init__(self, metric_keys:dict):
        self.metric_dict = {}
        self.register_keys(metric_keys)
    
    def register_keys(self, keys: list)->None:
        for key in keys:
            self.metric_dict[key] = []
    
    def load_metrics(self, csv_path: str, resume: bool, model_selection_criterion: str
                    ) -> Tuple[float, float, int]:
        
        resume_from_epoch = -1
        best_kappa        = -10
        best_loss         = 1000
        if not os.path.isfile(csv_path) or not resume:
            return best_kappa, best_loss, resume_from_epoch
        # if csv file exist, then first find out the epoch with best kappa(named resume_from_epoch), 
        # get the losses, kappa values within range 0~ best_epoch +1
        df = pd.read_csv(csv_path)
        for key in self.metric_dict.keys():
            if key not in df.columns:
                print(f"Key {key} not found in {df.columns}, not loading csv")
                return best_kappa, best_loss, resume_from_epoch

        test_criterion = list(df[model_selection_criterion])
        
        best_idx   = np.argmax(np.array(test_criterion))
        best_criterion = test_criterion[best_idx]
        best_loss  = min(list(df['test_losses'])[:best_idx+1])
        resume_from_epoch = best_idx+1

        for key in self.metric_dict.keys():
            self.metric_dict[key]= list(df[key])[:resume_from_epoch]

        print("================Loading CSV==================")
        print(f"|Loading csv from {csv_path},")
        print(f"|best test loss = {best_loss:.4f},")
        print(f"|best {model_selection_criterion}     = {best_criterion:.4f},")
        print(f"|epoch          = {resume_from_epoch:.4f}")
        print("=============================================")
        return best_criterion, best_loss, resume_from_epoch
    
    def save_metrics(self, csv_path: str, debug=False) -> None:
        df = pd.DataFrame(self.metric_dict)
        print(df)
        if not debug:
            df.to_csv(csv_path, index=False)
    
    def push_loss_acc_kappa(self, loss, acc, kappa, train=True):
        if train:
            self.metric_dict['train_losses'].append(loss)
            self.metric_dict['train_acc'].append(acc)
            self.metric_dict['train_kappa'].append(kappa)
        else: # test/valid
            self.metric_dict['test_losses'].append(loss)
            self.metric_dict['test_acc'].append(acc)
            self.metric_dict['test_kappa'].append(kappa)
    
    def push_auc_precision_recall_AP_f1(self, auc: np.float32, precision: np.float32, 
                                        recall:np.float32, AP: np.float32, f1: np.float32) -> None:
        self.metric_dict['auc'].append(auc)
        self.metric_dict['precision'].append(precision)
        self.metric_dict['recall'].append(recall)
        self.metric_dict['AP'].append(AP)
        self.metric_dict['f1'].append(f1)
    
    def print_summary(self, epoch, total_epoch, lr):
        print(f"[{epoch+1}/{total_epoch}] lr = {lr:.7f}, ", end='')
        for key in self.metric_dict.keys():
            print(f"{key} =  {self.metric_dict[key][epoch]}, ", end='')
        print('\n')
    
    def write_to_tensorboard(self, writer, epoch):
        for key in self.metric_dict.keys():
            writer.add_scalar(key, self.metric_dict[key][-1], epoch)
    
if __name__ == '__main__':
    m = Metric()
    m.load_metrics('../checkpoint/tmp.csv')

    for i in range(10):
        train_ = [np.random.rand() for i in range(3)]
        test_  = [np.random.rand() for i in range(3)]
        m.push_loss_acc_kappa(*train_, train=True)
        m.push_loss_acc_kappa(*test_,  train=False)
    
    m.save_metrics('../checkpoint/tmp.csv')
