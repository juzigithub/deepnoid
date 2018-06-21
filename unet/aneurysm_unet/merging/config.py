### Model info ###
# Densenet

### Path info ###

# Linux #
MODE = 'linux'
DATA_PATH = '/home/mspark/project/data/Brain_Aneurysm_new_dataset'
TRAIN_DATA_PATH = DATA_PATH + '/train'
VAL_DATA_PATH = DATA_PATH + '/test'
PKL_DATA_PATH = DATA_PATH + '/pkl'
PKL_NAME = 'train.pkl'

# Window #
# MODE = 'window'
# DATA_PATH = 'D:\\dataset\Brain_Aneurysm_new_dataset'
# TRAIN_DATA_PATH = DATA_PATH + '\\train'
# VAL_DATA_PATH = DATA_PATH + '\\test'
# PKL_DATA_PATH = DATA_PATH + '\\pkl'
# PKL_NAME = 'train.pkl'

###  File setting ###

DATA_FILE_TYPE = 'pkl'
REBUILD_PKL = False
PATH_SLASH = '/' if MODE == 'linux' else '\\'
IMG_SIZE = 256
VALIDATION_RATIO = 15

### Common ###

FIRST_DOWNSAMPLING = False
EPOCHS = 200
SAVING_EPOCH = 5
BATCH_SIZE = 28
BUFFER_SIZE = 3000
INIT_N_FILTER = 24
ACTIVATION_FUNC = 'relu'        # relu, lrelu, elu, prelu, selu
LOSS_FUNC = 'dice'              # dice, focal, cross_entropy, dice_sum, huber, weighted_cross_entropy
LAMBDA = 0.8                    # forground loss : LAMBDA / background loss : (1 - LAMBDA)
OPTIMIZER = 'adam'              # adam, rmsprop, sgd
INIT_LEARNING_RATE = 0.005
DECAY_RATE = 0.9
DECAY_STEP = 2500
DECAY_STAIRCASE = True
NORMALIZATION_TYPE = 'batch'    # batch, group
DOWNSAMPLING_TYPE = 'neighbor'   # neighbor, maxpool, avgpool
UPSAMPLING_TYPE = 'transpose'   # resize, transpose, add, concat
GROUP_N = 8                     # group size of group_conv & group_norm
DROPOUT_RATE = 0.5

### Unet ###
DEPTH = 4
RGB_CHANNEL = 1
N_CLASS = 2

### Densenet ###
GROWTH_RATE = 12
THETA = 0.5

### Mobilenet ###
WIDTH_MULTIPLIER = 1.0          # out_channel = in_channel * width_multiplier

### Shufflenet ###
UNIT_N = 2                      # number of shuffle unit

### HENet ###
GROUP_IN = GROUP_N              # number of groups in first group_conv
GROUP_OUT = 6                   # number of groups in last group_conv
REPEAT = 2                      # repeat * s1_block

### Result ###
PRED_MASKING_COLOR = 'red'      # red or green or blue
LABEL_MASKING_COLOR = 'blue'    # red or green or blue
W = 40
P = 0.0001
