### Model info ###

### Path info ###

## hshin ##
# Linux #
# MODE = 'linux'
# GPU = '1'
# SAVE_DATA_PATH = './npy/train/'        # train
## SAVE_DATA_PATH = './npy/validation/'  # validation
## SAVE_DATA_PATH = './npy/test/'        # test
# HGG_DATA_PATH = "/home/hshin255/MICCAI_BraTS_2018_Data_Training_for_test/HGG/"
# LGG_DATA_PATH = "/home/hshin255/MICCAI_BraTS_2018_Data_Training_for_test/LGG/"
# SURVIVAL_CSV_PATH = "/home/hshin255/MICCAI_BraTS_2018_Data_Training_for_test/survival_data.csv"

# Window #
# MODE = 'window'
# GPU = '1'
# SAVE_DATA_PATH = '.\\npy\\train\\'         # train
## SAVE_DATA_PATH = '.\\npy\\validation\\'   # validation
## SAVE_DATA_PATH = '.\\npy\\test\\'         # test
# HGG_DATA_PATH = "C:\\Users\\hshin\\Desktop\\MICCAI_BraTS_2018_Data_Training\\HGG"
# LGG_DATA_PATH = "C:\\Users\\hshin\\Desktop\\MICCAI_BraTS_2018_Data_Training\\LGG"
# SURVIVAL_CSV_PATH = "C:\\Users\\hshin\\Desktop\\MICCAI_BraTS_2018_Data_Training\\survival_data.csv"

## mspark ##
# Linux #
MODE = 'linux'
GPU = '6'
SAVE_DATA_PATH = './npy/train/'    # train
# SAVE_DATA_PATH = './npy/validation/' # validation
# SAVE_DATA_PATH = './npy/test/'       # test
HGG_DATA_PATH = '/home/mspark/project/data/brats2018/MICCAI_BraTS_2018_Data_Training/HGG/'
LGG_DATA_PATH = '/home/mspark/project/data/brats2018/MICCAI_BraTS_2018_Data_Training/LGG/'
VAL_DATA_PATH = '/home/mspark/project/data/brats2018/MICCAI_BraTS_2018_Data_Validation/'
SURVIVAL_CSV_PATH = '/home/mspark/project/data/brats2018/MICCAI_BraTS_2018_Data_Training/survival_data.csv'

# Window #
# MODE = 'window'
# GPU = '6'
# SAVE_DATA_PATH = '.\\npy\\train\\'        # train
## SAVE_DATA_PATH = '.\\npy\\validation\\'  # validation
## SAVE_DATA_PATH = '.\\npy\\test\\'        # test
# HGG_DATA_PATH = 'D:\\dataset\\BRATS\\2018\\small_data\\HGG\\'
# LGG_DATA_PATH = 'D:\\dataset\\BRATS\\2018\\small_data\\LGG\\'
# VAL_DATA_PATH = 'D:\\dataset\\BRATS\\2018\\MICCAI_BraTS_2018_Data_Validation\\'
# SURVIVAL_CSV_PATH = 'D:\\dataset\BRATS\\2018\\MICCAI_BraTS_2018_Data_Training\\survival_data.csv'


###  File setting ###
DATA_FILE_TYPE = 'npy'
REBUILD_DATA = True
TRAIN_YN = True if 'train' in SAVE_DATA_PATH else False
PATH_SLASH = '/' if MODE == 'linux' else '\\'
IMG_SIZE = 240
VALIDATION_RATIO = 15
SPLITS = 5
SUB_SPLITS = 3
RGB_CHANNEL = 1
N_CLASS = 4


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

### Resnet ###
RES_DEPTH = 3           # 3 for v2, 4 for v1
MODE = 'bottleneck_v2_with_unet'  # possible block : residual_block_v1, bottleneck_block_v1,residual_block_v2, bottleneck_block_v2
                        # possible mode :  bottleneck_v1_with_unet, bottleneck_v2_with_unet, residual_v1_with_unet, residual_v2_with_unet
                        # possible resnet : bottleneck_v1, residual_v1, bottleneck_v2, residual_v2
bottleneck = True
v2_depth = 92
choices = {
      18: [2, 2, 2, 2],        # recommends residual_block_v1
      34: [3, 4, 6, 3],        # recommends residual_block_v1
      50: [3, 4, 6, 3],        # bottleneck_block_v1
      101: [3, 4, 23, 3],     # bottleneck_block_v1
      152: [3, 8, 36, 3],     # bottleneck_block_v1
      200: [3, 24, 36, 3],    # bottleneck_block_v1
      108: [12, 12, 12],       # v2, n = (depth - 2) / 9, for bottleneck_block_v2
      164: [18, 18, 18],       # v2
      1001: [111, 111, 111],   # v2
      'another': [(v2_depth - 2) / 9, (v2_depth - 2) / 9,(v2_depth - 2) / 9]   #v2
  }
n_filter_chunks = choices[108]
# n_filter_chunks = [3,4,6,3]         # or assign numbers of channels you want
# n_filters = [16, 32, 64, 128]
INIT_FILTER = 8
kernel_size = 1
stride = [1,1]
n_blocks = 5
n_classes = 2
training = True





















'''
references

https://github.com/zsdonghao/u-net-brain-tumor/blob/master/prepare_data_with_valid.py#L18

'''