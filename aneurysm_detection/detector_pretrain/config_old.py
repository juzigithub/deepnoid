import numpy as np
### Model info ###

### Path info ###
# snu #
MODE = 'linux'
GPU = '6'
INPUT_PATH = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/input_dcm_original/'
TXT_PATH = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/label/'
DCM_PATH = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/input_dcm_original/*/*/*/'
LABEL_PATH = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/label/*/*/*/'
NPZ_PATH = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/npz/'

RESTORE = False                 # load weights file
PATH_SLASH = '/' if MODE == 'linux' else '\\'

### Data info ###
IMG_SIZE = [256, 256]         # axial : [192, 192], sagittal : [155, 192], coronal : [155, 192]
POOLED_SIZE = [8, 8]
N_3D_CONTEXT = 9
N_INPUT_CHANNEL = 3
# LANDMARK_DIVIDE = 10
# RGB_CHANNEL = 3
N_CLASS = 2
# TRAIN_LABEL = [0, 1, 2, 3]
# ET_LABEL = [0, 0, 0, 1]
# TC_LABEL = [0, 1, 0, 1]
# WT_LABEL = [0, 1, 1, 1]

### Common ###
EPOCHS = 100                     # epochs
# SPLITS = 5                      # cross validation cnt
SAVING_EPOCH = 1                # save model/img every SAVING_EPOCH
BATCH_SIZE = 1
INIT_N_FILTER = 48              # output n_channel(n_filter) of first conv layer
ACTIVATION_FUNC = 'elu'         # relu, lrelu, elu, prelu, selu
LOSS_FUNC = 'dice_sum'          # g_dice, dice, focal, cross_entropy, dice_sum, huber, weighted_cross_entropy
LAMBDA = [0.2, 0.8]   # weight of each loss [bg, ncr, ed, et]
OPTIMIZER = 'adam'           # adam, rmsprop, sgd
INIT_LEARNING_RATE = 1e-6
DECAY_RATE = 0.9
DECAY_STEP = 4000
DECAY_STAIRCASE = True
NORMALIZATION_TYPE = 'group'    # batch, group
PRETRAIN_N_LAYERS = 12
N_DOWNSAMPLING = 3
# N_LAYERS = [3, 3, 4]            # n_layers before each downsampling
# N_LAYERS_HIGH = [2,2,2,3]       # n_high_layers before each downsampling
# N_LAYERS_LOW = [2,3,3]          # n_low_layers before each downsampling
# DEPTH = len(N_LAYERS)           # total downsampling cnt. if 4 then img size(192 -> 96 -> 48 -> 24 -> 12)
# DEPTH_HIGH = len(N_LAYERS_HIGH)
# DEPTH_LOW = len(N_LAYERS_LOW)
# DOWNSAMPLING_TYPE = 'neighbor'  # neighbor, maxpool, avgpool
UPSAMPLING_TYPE = 'resize'         # resize, transpose, add, concat, avgpool
GROUP_N = 4                     # group size of group_conv & group_norm
INIT_DROPOUT_RATE = 0.2
DROPOUT_INCREASE_RATE = 1.00    # 1.11^10=2.8394

### Mobilenet ###
WIDTH_MULTIPLIER = 1.0          # out_channel = in_channel * width_multiplier

### Histogram Match ###
# BATCH_MATCH_THRES = 'fuzzy_log'         # 'fuzzy', 'mean', 'median', 'valley', fuzzy_log
# N_MATCH_DIVIDE = 10
# STANDARD = False
# SCALE = 1

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
DETECTION_MIN_CONFIDENCE = 0.3 #################### 0.7
DETECTION_MAX_INSTANCES = 10
DETECTION_NMS_THRESHOLD = 0.3
