### Path info ###

# lINUX #
# TRAIN_DATA_PATH = '/home/mspark/project/data/Brain_Aneurysm_new_dataset/train'
# VAL_DATA_PATH = '/home/mspark/project/data/Brain_Aneurysm_new_dataset/test'
# PKL_DATA_PATH = '/home/mspark/project/data/Brain_Aneurysm_new_dataset/pkl/'
# LOG_DATA_PATH = '/home/mspark/project/aneurysm/unet/results/'
# PKL_NAME = 'train.pkl'

# WINDOW #
TRAIN_DATA_PATH = 'D:\\dataset\\Brain_Aneurysm_new_dataset\\train'
VAL_DATA_PATH = 'D:\\dataset\\Brain_Aneurysm_new_dataset\\test'
PKL_DATA_PATH = 'D:\\dataset\\Brain_Aneurysm_new_dataset\\pkl'
LOG_DATA_PATH = '/home/mspark/project/aneurysm/unet/results/'
PKL_NAME = 'train.pkl'


### pkl setting ###

REBUILD_PKL = False             # pkl 파일 새로 만들지 결정 (pkl setting 바꿀 때 True)
MODE = 'window'                # 'window' or 'linux'
IMG_SIZE = 256
VALIDATION_RATIO = 15


### Common ###

# epoch, batch
EPOCHS = 200
SAVING_EPOCH = 2
BATCH_SIZE = 28

# learning_rate
INIT_LEARNING_RATE = 0.005
DECAY_RATE = 0.9
DECAY_STEP = 2500
DECAY_STAIRCASE = True

# dropout
DROPOUT_RATE = 0.2

### Unet ###

DEPTH = 4                      # Unet 층 깊이 (down * depth + up * depth)
INIT_N_FILTER = 32             # 첫 레이어의 필터수





