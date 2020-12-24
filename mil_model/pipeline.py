import time
import os
from tqdm import tqdm
import typing
from typing import Callable, Tuple
import numpy as np

import torch
from torch import nn
from torch.utils.data import dataloader

from .loss import kappa_metric, correct, flatten_list_tensor_to_numpy, get_auc_precision_recall_AP_f1
from .util import Metric

def train(cfg, model: nn.Module, optimizer: object, train_loader: dataloader, criterion: Callable, 
          loss_kappa_acc_metrics: object, epoch: int) -> None:
    # ===============================================================================================
    #                                 Train for loop block
    # ===============================================================================================
    total_loss    = 0.0
    train_correct = 0.
    predictions   = []
    groundtruth   = []
    pbar = tqdm(enumerate(train_loader, 0))
    
    # tracking data time and GPU time and print them on tqdm bar.
    end_time = time.time()

    model.train()
    for i, data in pbar:

        # get the inputs; data is a list of [inputs, labels], put them in cuda
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        data_time = time.time()-end_time # data time
        end_time = time.time()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # feed forward
        outputs = model(inputs)

        # compute loss and backpropagation
        loss = criterion(outputs, labels)
        loss.backward()
        loss = loss.detach()

        optimizer.step()
        gpu_time = time.time()-end_time # gpu time

        # collect statistics
        predictions.append(outputs.detach().cpu())
        groundtruth.append(labels.detach().cpu())

        running_loss  =  loss.item()
        total_loss    += running_loss
        train_correct += correct(predictions[-1], groundtruth[-1])

        pbar.set_postfix_str(f"[{epoch}/{cfg.MODEL.EPOCHS}] [{i+1}/{len(train_loader)} "+
                                f"training loss={running_loss:.4f}, data time = {data_time:.4f}, gpu time = {gpu_time:.4f}")
        end_time = time.time()

    groundtruth = flatten_list_tensor_to_numpy(groundtruth, cfg.DATASET.NUM_GRADE)
    predictions = flatten_list_tensor_to_numpy(predictions, cfg.DATASET.NUM_GRADE)
    loss_kappa_acc_metrics.push_loss_acc_kappa(loss=total_loss/len(train_loader), 
                                                acc=train_correct/(len(train_loader)*cfg.MODEL.BATCH_SIZE), 
                                                kappa=kappa_metric(groundtruth, predictions), 
                                                train=True,
                                                )
    
def validation(cfg, model: nn.Module, test_loader: dataloader, criterion: Callable, epoch: int, loss_kappa_acc_metrics: object, 
               writer: object =None, optimizer: object = None, save_pred: bool = False) -> None:
    # ===============================================================================================
    #                                     TEST for loop block
    # ===============================================================================================
    test_total_loss = 0.0
    predictions = []
    groundtruth = []
    model.eval()
    for batch_4tensors, labels in tqdm(test_loader):
        #===============================================================================
        #    batch_4tensors is 4-tensor with shape (batch_size//4, 4, 3, 1536, 1536), 
        #    labels is a "single" tensor (batch_size, 5)
        #===============================================================================
        # 
        # The first dimension of the batch_4tensors is batch_size//4  because I take 4 images from each test slide 
        # and will ensemble the result.
        # Firstly, reshape the image tensor by combining the first two channel 
        #          (batch_size//4, 4, 3, 1536, 1536)--> (batch_size, 3, 1536, 1536)
        # After the tensor passes the model, I will resume the first two dimension of the output
        # to a (batch_size//4, 4, 5) tensor, and average the results by axis=1 --> get a shape (batch_size//4, 5)
        # output, similar to labels
        with torch.no_grad():
            first_two_dim = batch_4tensors.shape[:2]
            batch_tensors = batch_4tensors.view(-1, *batch_4tensors.shape[2:]).contiguous().cuda()
            labels        = labels.cuda()
            outputs       = model(batch_tensors)
            outputs       = torch.mean(outputs.view((*first_two_dim, -1)), axis=1)
            test_loss     = criterion(outputs, labels).item()
            test_total_loss += test_loss

            predictions.append(outputs.cpu())
            groundtruth.append(labels.cpu())
    
    groundtruth = flatten_list_tensor_to_numpy(groundtruth, cfg.DATASET.NUM_GRADE)
    predictions = flatten_list_tensor_to_numpy(predictions, cfg.DATASET.NUM_GRADE)

    if save_pred:
        gt_pred_arr = np.stack([groundtruth, predictions], axis=0)
        with open(cfg.MODEL.CHECKPOINT_PATH + 'prediction_arr_valid.npy', 'wb') as f:
            np.save(f, gt_pred_arr)

    kappa = kappa_metric(groundtruth, predictions)
    test_epoch_loss = test_total_loss/len(test_loader)
    loss_kappa_acc_metrics.push_loss_acc_kappa(loss=test_epoch_loss,
                                                acc=correct(groundtruth, predictions)/(predictions.shape[0]), 
                                                kappa=kappa, train=False,
                                                )
    
    loss_kappa_acc_metrics.push_auc_precision_recall_AP_f1(*get_auc_precision_recall_AP_f1(groundtruth, predictions))
    loss_kappa_acc_metrics.print_summary(epoch=epoch, total_epoch=cfg.MODEL.EPOCHS, 
                                         lr= optimizer.param_groups[0]['lr'] if optimizer else -1)
    
    if cfg.SOURCE.TENSORBOARD and writer:
        loss_kappa_acc_metrics.write_to_tensorboard(writer=writer, epoch=epoch)
        writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)

def test(cfg, test_loader: dataloader, model: nn.Module, criterion: Callable)->None:
    checkpoint_prefix = f"{cfg.MODEL.BACKBONE}_{cfg.DATASET.TILE_SIZE}_{cfg.DATASET.PATCH_SIZE}"
    
    # prepare resnet50, resume from checkpoint
    print("==============Building model=================")
    best_loss_path = os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+'best_loss.pth')
    model.resume_from_path(best_loss_path)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs...")
        model = nn.DataParallel(model)
    model = model.cuda()

    # prepare training and testing loss
    loss_kappa_acc_metrics = Metric([key for key in cfg.METRIC.KEYS if 'train' not in key])
    csv_path               = os.path.join(cfg.MODEL.CHECKPOINT_PATH, checkpoint_prefix+"_test_loss.csv")
    
     # test pipeline
    validation(cfg, model, test_loader, criterion, 0, loss_kappa_acc_metrics, save_pred=True)
    loss_kappa_acc_metrics.save_metrics(csv_path)
