import os
import time
from tqdm import tqdm

# torch library
import torch
from torch.utils.data import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
from torch import nn

# customized libraries
from mil_model.dataloader import TileDataset, PathoAugmentation, get_resnet_preproc_fn, VahadaneWrapper
from mil_model.config import get_cfg_defaults
from mil_model.resnet_model import CustomModel, build_optimizer
from mil_model.loss import get_bceloss, Callback, kappa_metric, correct, flatten_list_tensor_to_numpy, get_auc_precision_recall_AP_f1
from mil_model.util import Metric
from mil_model.pipeline import train, validation, test

if __name__ == "__main__":
    # get config variables
    cfg = get_cfg_defaults()

    # assign GPU
    device = ','.join(str(i) for i in cfg.SYSTEM.DEVICES)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    #torch.backends.cudnn.enabled = False
    #torch.backends.cudnn.benchmark = True

    # prepare Vahadane stain normalizer
    stain_normalizer= VahadaneWrapper(img_dir=cfg.DATASET.TRAIN_IMG_DIR ,
                                      csv1=cfg.DATASET.CSV1,
                                      csv2=cfg.DATASET.CSV2,
                                      vahadane1=cfg.DATASET.VAHADANE1, 
                                      vahadane2=cfg.DATASET.VAHADANE2,
                                     )
    
    # prepare train, test dataloader
    train_dataset = TileDataset(img_dir   =cfg.DATASET.TRAIN_IMG_DIR, 
                                json_dir  =cfg.DATASET.TRAIN_BBOX, 
                                patch_size=cfg.DATASET.PATCH_SIZE, 
                                patch_dir =cfg.DATASET.TRAIN_PATCH_DIR,
                                resize_ratio=cfg.DATASET.RESIZE_RATIO,
                                tile_size =cfg.DATASET.TILE_SIZE, 
                                is_test   =False,
                                aug       =PathoAugmentation, 
                                preproc   =get_resnet_preproc_fn(),
                                debug     = cfg.SOURCE.DEBUG,
                                balance   = True,
                                #stain_wrapper = stain_normalizer,
                                )
    valid_dataset  = TileDataset(img_dir    =cfg.DATASET.VALID_IMG_DIR, 
                                json_dir   =cfg.DATASET.VALID_BBOX, 
                                patch_size =cfg.DATASET.PATCH_SIZE, 
                                resize_ratio=cfg.DATASET.RESIZE_RATIO,
                                patch_dir =cfg.DATASET.VALID_PATCH_DIR,
                                tile_size  =cfg.DATASET.TILE_SIZE, 
                                is_test    =True,
                                aug        =None, 
                                preproc    =get_resnet_preproc_fn(), 
                                debug      = cfg.SOURCE.DEBUG,
                                balance    = True,
                                #stain_wrapper= stain_normalizer
                                )
                        
    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg.MODEL.BATCH_SIZE, 
                              shuffle=True , 
                              num_workers=4,
                              drop_last=True)
    # NOTE: the batch size of test loader is 4 times smaller than traning loader
    #       since test dataset returns a list of "4" image tensor.
    # However, the batch size does not really matter if the training resources is enough to 
    # accommodate 4 times the batch size of training loader.
    valid_loader  = DataLoader(valid_dataset , 
                              batch_size=cfg.MODEL.BATCH_SIZE//4, 
                              shuffle=False, 
                              num_workers=4,
                              drop_last=True
                              )

    # prepare for checkpoint info and callback
    if not os.path.isdir(cfg.MODEL.CHECKPOINT_PATH):
        os.makedirs(cfg.MODEL.CHECKPOINT_PATH)
    checkpoint_prefix = f"{cfg.MODEL.BACKBONE}_{cfg.DATASET.TILE_SIZE}_{cfg.DATASET.PATCH_SIZE}"
    early_stopping_callback = Callback(model_selection_criterion=cfg.METRIC.MODEL_SELECTION_CRITERION, 
                                       checkpoint_prefix=checkpoint_prefix)

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

    # prepare optimizer: Adam is suggested in this case.
    warmup_epo    = 3
    n_epochs      = cfg.MODEL.EPOCHS
    optimizer = build_optimizer(type=cfg.MODEL.OPTIMIZER, 
                                model=model, 
                                lr=cfg.MODEL.LEARNING_RATE)

    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=3e-6, T_max=(n_epochs-warmup_epo))
    scheduler = GradualWarmupScheduler(optimizer, 
                                       multiplier=1.0, 
                                       total_epoch=warmup_epo,
                                       after_scheduler=base_scheduler)

    # criterion: BCE loss for multilabel tensor with shape (BATCH_SIZE, 5)
    criterion = get_bceloss()

    # prepare tensorboard writer
    if cfg.SOURCE.TENSORBOARD:
        writer = SummaryWriter()
    
    # prepare training and testing loss
    loss_kappa_acc_metrics = Metric(cfg.METRIC.KEYS)
    csv_path               = os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"loss.csv")
    best_criterion, best_loss, resume_from_epoch = loss_kappa_acc_metrics.load_metrics(csv_path, 
                                                resume=cfg.MODEL.LOAD_CSV, 
                                                model_selection_criterion=cfg.METRIC.MODEL_SELECTION_CRITERION)
    early_stopping_callback.update_best(best_loss, best_criterion)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optimizer.zero_grad()
    optimizer.step()
    
    # training pipeline
    print("==============Start training=================")
    for epoch in range(0, cfg.MODEL.EPOCHS):  # loop over the dataset multiple times
        # update scheduler  
        if epoch < resume_from_epoch:
            scheduler.step()
            optimizer.step()
            continue
        scheduler.step()
        
        train(cfg, model, optimizer, train_loader, criterion, loss_kappa_acc_metrics, epoch)
        
        validation(cfg, model, valid_loader, criterion, epoch, loss_kappa_acc_metrics)
        
        if early_stopping_callback.on_epoch_end(cfg=cfg, metrics=loss_kappa_acc_metrics, csv_path=csv_path, model=model):
            break
    
    print('Finished Training')
    

    # Test block
    print("==============Start testing==================")
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
                              batch_size=cfg.MODEL.BATCH_SIZE, #cfg.MODEL.BATCH_SIZE//4, 
                              shuffle=False, 
                              num_workers=4,
                              drop_last=False
                              )

    test(cfg, test_loader, model, criterion)
    
    print('Finished Testing')