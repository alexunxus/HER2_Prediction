import os
import numpy as np
import torch
from torch.nn import BCELoss
from torch import nn
from sklearn.metrics import cohen_kappa_score, confusion_matrix, average_precision_score, auc, roc_curve

from yacs.config import CfgNode
import typing
from typing import Tuple

def get_bceloss():
    # binary cross entropy loss
    return BCELoss()

def kappa_score(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    # quadratic weighted kappa
    return cohen_kappa_score(gt, pred, weights='quadratic')

@torch.no_grad()
def kappa_metric(gts: np.ndarray, preds: np.ndarray) -> np.ndarray:
    '''
    gts:   nparray with shape (N, 5)
    preds: nparray with shape (N, 5)
    '''
    #gts   = np.concatenate([tensor.numpy()>0.5 for tensor in gts  ], axis=0).reshape((-1, last_dim))
    #preds = np.concatenate([tensor.numpy()>0.5 for tensor in preds], axis=0).reshape((-1, last_dim))
    gts   = gts.round().astype(np.int32).sum(1)
    preds = preds.round().astype(np.int32).sum(1)
    k     = kappa_score(gts, preds)
    conf  = confusion_matrix(gts, preds)
    print(f"Kappa score = {k}")
    print("Confusion matrix:\n", conf)
    return k

def correct(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    '''
    gt    is a (B, 5) tensor or [tensor[[batch, 5]...], tensor[[batch, 5]...]...]
    label is a (B, 5) tensor or [tensor[[batch, 5]...], tensor[[batch, 5]...]...]
    '''
    #if isinstance(gt, list):
    #    gt   = np.concatenate([tensor.numpy()>0.5 for tensor in gt  ], axis=0).reshape((-1, last_dim))
    #    pred = np.concatenate([tensor.numpy()>0.5 for tensor in pred], axis=0).reshape((-1, last_dim))
    if torch.is_tensor(gt):
        gt = gt.numpy()
        pred = pred.numpy()
    gt   = gt.round().astype(np.int32).sum(1) >= 3
    pred = pred.round().astype(np.int32).sum(1) >= 3
    return (gt == pred).sum()

def get_auc_precision_recall_AP_f1(gt: np.ndarray, pred: np.ndarray) -> \
        Tuple[np.float32, np.float32, np.float32, np.float32]:
    gt   = (gt.round()  .astype(np.int32).sum(1) >= 3).astype(np.int32)
    #pred = (pred.round().astype(np.int32).sum(1) >= 3).astype(np.int32)
    pred = pred[:, 2].astype(np.float32)
    AP   = average_precision_score(gt, pred)

    fpr, tpr, thresholds = roc_curve(gt, pred, pos_label=1)
    auc_ = auc(fpr, tpr)

    Tp = np.logical_and(gt == 1, pred >= 0.5).sum()
    Fp = np.logical_and(gt == 0, pred >= 0.5).sum()
    Fn = np.logical_and(gt == 1, pred <= 0.5).sum()

    precision = Tp/(Tp+Fp+1e-7)
    recall    = Tp/(Tp+Fn+1e-7)
    f1        = 2*precision*recall/(precision + recall + 1e-7)
    return   auc_, precision, recall, AP, f1

def flatten_list_tensor_to_numpy(list_tensor: list, last_dim: int) -> np.ndarray:
    return np.concatenate([tensor.numpy() for tensor in list_tensor], axis=0).reshape((-1, last_dim))

class Callback:
    def __init__(self, model_selection_criterion: str, checkpoint_prefix:str) -> None:
        self.model_selection_criterion = model_selection_criterion
        self.checkpoint_prefix = checkpoint_prefix
        self.patience=0
        self.best_loss=100
        self.best_criterion=-100
    
    def update_best(self, best_loss: np.ndarray, best_criterion: np.ndarray) -> None:
        self.best_criterion = best_criterion
        self.best_loss = best_loss

    def on_epoch_end(self, cfg: CfgNode, metrics: object, csv_path: str, model:nn.Module) -> int:
        # ===============================================================================================
        #                                     Callback block
        # ===============================================================================================
        # 1. Save best weight: 
        #   If for one epoch, the test loss or kappa is better than current best kappa and best loss, then
        #   I reset the patience and save the model
        #   Otherwise, patience <- patience+1, and the model weight is not saved.
        # 2. Early stopping: 
        #   If the patience >= the patience limit, then break the epoch for loop and finish training.
        # 3. Saving training curve:
        #   For each epoch, update the loss, kappa dictionary and save them.
        update_loss      = False
        update_criterion = False
        exit_signal = 0

        if cfg.SOURCE.DEBUG:
            print("Debugging, not saving...")
            metrics.save_metrics(csv_path=csv_path, debug=True)
        else:
            print('======Saving training curves=======')
            metrics.save_metrics(csv_path=csv_path, debug=False)
        
        test_epoch_loss = metrics.metric_dict['test_losses'][-1]
        if test_epoch_loss < self.best_loss:
            self.best_loss = test_epoch_loss
            if not cfg.SOURCE.DEBUG:
                torch.save(model.state_dict(), os.path.join(cfg.MODEL.CHECKPOINT_PATH, self.checkpoint_prefix+"best_loss.pth"))
            self.patience = 0
            update_loss = True
        criterion = metrics.metric_dict[self.model_selection_criterion][-1]
        if criterion >= self.best_criterion:
            self.best_criterion = criterion
            if not cfg.SOURCE.DEBUG:
                torch.save(model.state_dict(), 
                           os.path.join(cfg.MODEL.CHECKPOINT_PATH, 
                           self.checkpoint_prefix+f"best_{self.model_selection_criterion}.pth"))
            self.patience = 0
            update_criterion = True
        if not update_loss and not update_criterion:
            self.patience += 1
            print(f"Patience = {self.patience}")
            if self.patience >= cfg.MODEL.PATIENCE:
                print(f"Early stopping at epoch {len(metrics.metric_dict[self.model_selection_criterion])}")
                exit_signal = 1

        print(f"best loss={self.best_loss}, best {self.model_selection_criterion}={self.best_criterion}")
        return exit_signal


if __name__ == '__main__':
    a = np.array([1, 3, 1, 2, 2, 1, 1, 3, 1])
    b = np.array([1, 2, 1, 1, 1, 3, 3, 1, 2])
    print(kappa_score(a, b))
    print(kappa_score(b, a))

    test_array = [torch.rand(32, 5) for i in range(10)]
    test_array = np.array([tensor.numpy()>0.5 for tensor in test_array]).reshape((-1, 5))
    print(test_array)
    test_array = test_array.round().astype(np.int32).sum(1)
    print(test_array)
    
    with torch.no_grad():
        preds = [torch.rand(32, 5) for i in range(10)]
        gts   = [torch.rand(32, 5) for i in range(10)]
        gts = preds
        #kappa_metric(gts, preds)
        #kappa_metric(gts, gts)
    
    # BECLoss test
    a = torch.rand(32, 5)
    b = torch.rand(32, 5)
    print(a, b)
    criterion = get_bceloss()

    loss = criterion(a, b)
    print(loss)

    # 
    a = torch.tensor([1, 0, 0, 0, 1, 5, 0, 0, 4, 5])
    b = torch.tensor([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    print(confusion_matrix(a, b))
    print(kappa_score(a, b))
    
    # check correctedness
    a = torch.rand(32, 5)
    b = torch.rand(32, 5)
    with torch.no_grad():
        print(correct(a, b))
    
    # check f1 acc precision 
    a = np.random.rand(32, 5)
    b = np.random.rand(32, 5)
    print(get_auc_precision_recall_AP_f1(a, b))

