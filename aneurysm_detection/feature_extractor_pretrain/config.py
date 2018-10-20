import numpy as np
### Model info ###

### Path info ###
# snu #
# MODE = 'linux'
# GPU = '6'
# INPUT_PATH = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/input_dcm_original/'
# TXT_PATH = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/label/'
# DCM_PATH = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/input_dcm_original/*/*/*/'
# LABEL_PATH = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/label/*/*/*/'
# NPZ_PATH = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/npz/'
# yonsei #
MODE = 'linux'
GPU = '6'
# INPUT_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/input/'
# LABEL_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/input'
NPZ_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/total/'

TEST_INPUT_PATH = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/test/input_dcm_original/'
TEST_LABEL_PATH = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/test/label/'
TEST_NPZ_PATH = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/test/npz/'

###  File setting ###
RESTORE = False                 # load weights file
PATH_SLASH = '/' if MODE == 'linux' else '\\'

### Data info ###
IMG_SIZE = [256, 256]         # axial : [192, 192], sagittal : [155, 192], coronal : [155, 192]
POOLED_SIZE = [8, 8]
N_3D_CONTEXT = 9
N_INPUT_CHANNEL = 3
N_CLASS = 2

### Common ###
EPOCHS = 100                     # epochs
SAVING_EPOCH = 1                # save model/img every SAVING_EPOCH
BATCH_SIZE = 1
INIT_N_FILTER = 48              # output n_channel(n_filter) of first conv layer
ACTIVATION_FUNC = 'elu'         # relu, lrelu, elu, prelu, selu
LAMBDA = [0.2, 0.8]   # weight of each loss [bg, ncr, ed, et]
OPTIMIZER = 'rmsprop'           # adam, rmsprop, sgd
INIT_LEARNING_RATE = 8e-7
DECAY_RATE = 0.9
DECAY_STEP = 100000
DECAY_STAIRCASE = True
NORMALIZATION_TYPE = 'group'    # batch, group
PRETRAIN_N_LAYERS = 15
N_DOWNSAMPLING = 3
N_EACH_DOWN_LAYERS = [3, 3, 4, 5]
UPSAMPLING_TYPE = 'resize'         # resize, transpose, add, concat, avgpool
GROUP_N = 4                     # group size of group_conv & group_norm
INIT_DROPOUT_RATE = 0.2
DROPOUT_INCREASE_RATE = 1.00    # 1.11^10=2.8394

### Rpn ###
ANCHOR_SCALES = [8, 16, 32, 64]     #   [8, 16, 32] <- [int(IMAGE_WIDTH / i) for i in [32, 16, 8, 4, 2]]
ANCHOR_RATIOS = [0.5, 1, 1.5, 2]
FEATURE_STRIDES = [8]
ANCHOR_STRIDE = 1
RPN_N_FILTER = 512
IMAGES_PER_GPU = 1

RPN_TRAIN_ANCHORS_PER_IMAGE = 256
RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

POST_NMS_ROIS_TRAINING = 2000
POST_NMS_ROIS_INFERENCE = 1000
RPN_NMS_THRESHOLD = 0.7

### detector ###
TRAIN_ROIS_PER_IMAGE = 200
ROI_POSITIVE_RATIO = 0.33
DETECTION_MIN_CONFIDENCE = 0.9 #################### 0.7
DETECTION_MAX_INSTANCES = 10
DETECTION_NMS_THRESHOLD = 0.3
