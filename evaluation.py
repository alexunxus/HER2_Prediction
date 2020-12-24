import os
import numpy as np 
import pandas as pd
from tqdm import tqdm
import time

# torch library
import torch
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
from torch import nn

# customized libraries
from mil_model.dataloader import TileDataset, PathoAugmentation, get_resnet_preproc_fn
from mil_model.config import get_cfg_defaults
from mil_model.resnet_model import CustomModel, build_optimizer
from mil_model.loss import get_bceloss
from mil_model.util import Metric
from mil_model.pipeline import validation


if __name__ == '__main__':
    # get config variables
    cfg = get_cfg_defaults()

    # assign GPU
    device = ','.join(str(i) for i in cfg.SYSTEM.DEVICES)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    # prepare train, test dataloader
    test_dataset  = TileDataset(img_dir    =cfg.DATASET.TEST_IMG_DIR, 
                                json_dir   =cfg.DATASET.TEST_BBOX, 
                                patch_size =cfg.DATASET.PATCH_SIZE, 
                                patch_dir  =cfg.DATASET.TEST_PATCH_DIR,
                                resize_ratio=cfg.DATASET.RESIZE_RATIO,
                                tile_size  =cfg.DATASET.TILE_SIZE, 
                                is_test    =True,
                                aug        =None, 
                                preproc    =get_resnet_preproc_fn(), 
                                debug      = cfg.SOURCE.DEBUG)
                        
    test_loader  = DataLoader(test_dataset , 
                              batch_size=1, #cfg.MODEL.BATCH_SIZE//4, 
                              shuffle=False, 
                              num_workers=4,
                              drop_last=True
                              )

    # prepare for checkpoint info and callback
    os.makedirs(cfg.MODEL.CHECKPOINT_PATH, exist_ok=True)
    checkpoint_prefix = f"{cfg.MODEL.BACKBONE}_{cfg.DATASET.TILE_SIZE}_{cfg.DATASET.PATCH_SIZE}"
    
    # prepare resnet50, resume from checkpoint
    print("==============Building model=================")
    model = CustomModel(backbone=cfg.MODEL.BACKBONE, 
                        num_grade=cfg.DATASET.NUM_GRADE, 
                        resume_from=cfg.MODEL.RESUME_FROM,
                        norm=cfg.MODEL.NORM_USE)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs...")
        model = nn.DataParallel(model)
    model = model.cuda()

    # criterion: BCE loss for multilabel tensor with shape (BATCH_SIZE, 5)
    criterion = get_bceloss()

    # prepare tensorboard writer
    if cfg.SOURCE.TENSORBOARD:
        writer = SummaryWriter()
    
    # prepare training and testing loss
    loss_kappa_acc_metrics = Metric([key for key in cfg.METRIC.KEYS if 'train' not in key])
    csv_path               = os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"_test_loss.csv")
    
     # training pipeline
    print("==============Start testing==================")
    validation(cfg, model, test_loader, criterion, 0, loss_kappa_acc_metrics, save_pred=True)
    loss_kappa_acc_metrics.save_metrics(csv_path)
    
    print('Finished Testing')
