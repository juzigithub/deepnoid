### Model info ###

### Path info ###

## hshin ##
# Linux #
# MODE = 'linux'
# GPU = '1'
# SAVE_TRAIN_DATA_PATH = './npy/train/'        # train
# SAVE_VALID_DATA_PATH = './npy/validation/'  # validation
# SAVE_TEST_DATA_PATH = './npy/test/'        # test
# HGG_DATA_PATH = "/home/hshin255/MICCAI_BraTS_2018_Data_Training_for_test/HGG/"
# LGG_DATA_PATH = "/home/hshin255/MICCAI_BraTS_2018_Data_Training_for_test/LGG/"
# SURVIVAL_CSV_PATH = "/home/hshin255/MICCAI_BraTS_2018_Data_Training_for_test/survival_data.csv"
# VAL_SURVIVAL_CSV_PATH = '####1231231############PATH_TO/survival_evaluation.csv##############123123########'

# Window #
# MODE = 'window'
# GPU = '1'
# SAVE_TRAIN_DATA_PATH = '.\\npy\\train\\'         # train
# SAVE_VALID_DATA_PATH = '.\\npy\\validation\\'   # validation
# SAVE_TEST_DATA_PATH = '.\\npy\\test\\'         # test
# HGG_DATA_PATH = "C:\\Users\\hshin\\Desktop\\MICCAI_BraTS_2018_Data_Training\\HGG"
# LGG_DATA_PATH = "C:\\Users\\hshin\\Desktop\\MICCAI_BraTS_2018_Data_Training\\LGG"
# SURVIVAL_CSV_PATH = "C:\\Users\\hshin\\Desktop\\MICCAI_BraTS_2018_Data_Training\\survival_data.csv"
# VAL_SURVIVAL_CSV_PATH = '#########################PATH_TO\\survival_evaluation.csv#####################3'

## mspark ##
# Linux #
MODE = 'linux'
GPU = '6'
SAVE_TRAIN_DATA_PATH = '/mnt/sdb/mspark/data/brats2018/npy/train/'      # train
SAVE_VALID_DATA_PATH = '/mnt/sdb/mspark/data/brats2018/npy/validation/' # validation
SAVE_TEST_DATA_PATH = '/mnt/sdb/mspark/data/brats2018/npy/test/'        # test
SAVE_SURVIVAL_DATA_PATH = '/mnt/sdb/mspark/data/brats2018/npy/task2/'
HGG_DATA_PATH = '/mnt/sdb/mspark/data/brats2018/MICCAI_BraTS_2018_Data_Training/HGG/'
LGG_DATA_PATH = '/mnt/sdb/mspark/data/brats2018/MICCAI_BraTS_2018_Data_Training/LGG/'
VAL_DATA_PATH = '/mnt/sdb/mspark/data/brats2018/MICCAI_BraTS_2018_Data_Validation/'
SURVIVAL_CSV_PATH = '/mnt/sdb/mspark/data/brats2018/MICCAI_BraTS_2018_Data_Training/survival_data.csv'
VAL_SURVIVAL_CSV_PATH = '/mnt/sdb/mspark/data/brats2018/MICCAI_BraTS_2018_Data_Validation/survival_evaluation.csv'

# Window #
# MODE = 'window'
# GPU = '6'
# SAVE_TRAIN_DATA_PATH = '.\\npy\\train\\'        # train
# SAVE_VALID_DATA_PATH = '.\\npy\\validation\\'  # validation
# SAVE_TEST_DATA_PATH = '.\\npy\\test\\'        # test
# HGG_DATA_PATH = 'D:\\dataset\\BRATS\\2018\\small_data\\HGG\\'
# LGG_DATA_PATH = 'D:\\dataset\\BRATS\\2018\\small_data\\LGG\\'
# VAL_DATA_PATH = 'D:\\dataset\\BRATS\\2018\\MICCAI_BraTS_2018_Data_Validation\\'
# SURVIVAL_CSV_PATH = 'D:\\dataset\BRATS\\2018\\MICCAI_BraTS_2018_Data_Training\\survival_data.csv'
# VAL_SURVIVAL_CSV_PATH = 'D:\\dataset\BRATS\\2018\\MICCAI_BraTS_2018_Data_Validation\\survival_evaluation.csv'

###  File setting ###
DATA_FILE_TYPE = 'npy'
REBUILD_HM_DATA = False
REBUILD_TASK1_DATA = False      # train_mobile.py
REBUILD_TASK2_DATA = False      # train_mobile.py
REBUILD_TASK1_VAL_DATA = False
REBUILD_TASK2_VAL_DATA = False
RESTORE = False                # load weights file
PATH_SLASH = '/' if MODE == 'linux' else '\\'
N_FILES = 5

### Data info ###
IMG_SIZE = [192, 160]
PATCH_SIZE = 64
PATCH_STRIDE = 16
PATCH_NCR_CUTLINE = 0.05
PATCH_WT_CUTLINE = 0.25
VAL_PATCH_RATIO = 0.1
N_PATCH_TO_IMG = (((IMG_SIZE[0] - PATCH_SIZE) // PATCH_STRIDE) + 1) * (((IMG_SIZE[1] - PATCH_SIZE) // PATCH_STRIDE) + 1)
print(N_PATCH_TO_IMG)
USED_MODALITY = ['flair', 't1', 't1ce', 't2']
N_INPUT_CHANNEL = len(USED_MODALITY)
RGB_CHANNEL = 1
N_CLASS = 4
TRAIN_LABEL = [0, 1, 2, 3]
ET_LABEL = [0, 0, 0, 1]
TC_LABEL = [0, 1, 0, 1]
WT_LABEL = [0, 1, 1, 1]

### Common ###
EPOCHS = 10                     # epochs
SPLITS = 5                      # cross validation cnt
SAVING_EPOCH = 1                # save model/img every SAVING_EPOCH
BATCH_SIZE = 100
INIT_N_FILTER = 48              # output n_channel(n_filter) of first conv layer
ACTIVATION_FUNC = 'elu'         # relu, lrelu, elu, prelu, selu
LOSS_FUNC = 'dice'              # dice, focal, cross_entropy, dice_sum, huber, weighted_cross_entropy
LAMBDA = [0.01, 0.4, 0.2, 0.4]   # weight of each loss [bg, ncr, ed, et]
OPTIMIZER = 'rmsprop'           # adam, rmsprop, sgd
INIT_LEARNING_RATE = 5e-4
DECAY_RATE = 0.9
DECAY_STEP = 2000
DECAY_STAIRCASE = True
NORMALIZATION_TYPE = 'batch'    # batch, group
N_LAYERS = [5, 5, 5]            # n_layers before each downsampling
DEPTH = len(N_LAYERS)           # total downsampling cnt. if 4 then img size(192 -> 96 -> 48 -> 24 -> 12)
DOWNSAMPLING_TYPE = 'neighbor'  # neighbor, maxpool, avgpool
UPSAMPLING_TYPE = 'add'         # resize, transpose, add, concat, avgpool
GROUP_N = 4                     # group size of group_conv & group_norm
DROPOUT_RATE = 0.5

### Mobilenet ###
WIDTH_MULTIPLIER = 1.0          # out_channel = in_channel * width_multiplier