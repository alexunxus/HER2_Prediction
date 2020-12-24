from yacs.config import CfgNode as CN

_C                = CN() # Node, lv0
_C.SYSTEM         = CN() # None, lv1
_C.SYSTEM.DEVICES = [5]

_C.SOURCE = CN()
_C.SOURCE.DEBUG       = False
_C.SOURCE.TENSORBOARD = False

_C.DATASET = CN()
_C.DATASET.TRAIN_BBOX      = './train_bbox/data_2048/'
_C.DATASET.VALID_BBOX      = './valid_bbox/data_2048/'
_C.DATASET.TEST_BBOX       = './test_bbox/data_2048/'
_C.DATASET.TRAIN_IMG_DIR   = './train_images/'
_C.DATASET.VALID_IMG_DIR   = './valid_images/'
_C.DATASET.TEST_IMG_DIR    = './test_images/'
_C.DATASET.TRAIN_PATCH_DIR = './train_patches/'
_C.DATASET.VALID_PATCH_DIR = './valid_patches/'
_C.DATASET.TEST_PATCH_DIR  = './test_patches/'
_C.DATASET.CSV1            = '/mnt/cephrbd/data/A20009_CGMH_HER2/HER2_2016-2019_0323.csv'
_C.DATASET.CSV2            = '/mnt/cephrbd/data/A20009_CGMH_HER2/HER2_positive_2016-2019_compact.csv'
_C.DATASET.VAHADANE1       = './checkpoint/vahadane1.npy'
_C.DATASET.VAHADANE2       = './checkpoint/vahadane2.npy'

                               

_C.DATASET.NUM_GRADE       = 4
_C.DATASET.PATCH_SIZE      = 2048
_C.DATASET.TILE_SIZE       = 6 # 6
_C.DATASET.RESIZE_RATIO    = 4 # 8

_C.MODEL = CN()
_C.MODEL.BACKBONE          = 'R-18' #'baseline', 'R-50-xt', 'R-50-st', 'enet-b0', 'enet-b1'
_C.MODEL.BATCH_SIZE        = 7
_C.MODEL.EPOCHS            = 30
_C.MODEL.LEARNING_RATE     = 1e-5
_C.MODEL.USE_PRETRAIN      = True
_C.MODEL.NORM_USE          = "bn" # bn, gn, dn
_C.MODEL.OPTIMIZER         = "Adam" #"SGD" # SGD, Adam
_C.MODEL.CRITERION         = "BCE"
_C.MODEL.CHECKPOINT_PATH   = './checkpoint/'
_C.MODEL.RESUME_FROM       = './checkpoint/R-18_6_2048best_loss.pth' #'./checkpoint/baseline_6_2048best_f1.pth'
_C.MODEL.LOAD_CSV          = True
_C.MODEL.PATIENCE          = 5

_C.METRIC = CN()
_C.METRIC.KEYS = ['test_kappa','train_kappa','test_acc','train_acc',
                  'test_losses','train_losses', 'auc', 'precision','recall', 'AP', 'f1']
_C.METRIC.MODEL_SELECTION_CRITERION = 'auc'

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()