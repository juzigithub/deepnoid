### Path info ###

# lINUX #
MODE = 'linux'
DATA_PATH = '/home/mspark/project/data/Brain_Aneurysm_new_dataset'
TRAIN_DATA_PATH = DATA_PATH + '/train'
VAL_DATA_PATH = DATA_PATH + '/test'
PKL_DATA_PATH = DATA_PATH + '/pkl'
LOG_DATA_PATH = '/home/mspark/project/aneurysm/unet/results/'
PKL_NAME = 'train.pkl'

# WINDOW #
# MODE = 'window'
# DATA_PATH = 'D:\\dataset\\Brain_Aneurysm_new_dataset'
# TRAIN_DATA_PATH = DATA_PATH + '\\train'
# VAL_DATA_PATH = DATA_PATH + '\\test'
# PKL_DATA_PATH = DATA_PATH + '\\pkl'
# LOG_DATA_PATH = 'C:\\Users\\sunki\\PycharmProjects\\deepnoid\\Unet\\aneurysm_unet\\merging\\'
# PKL_NAME = 'train.pkl'

### file setting ###

DATA_FILE_TYPE = 'pkl'         # 데이터 로드 및 세이브 형식 결정 (pkl or json) json 은 차후 구현
REBUILD_PKL = True             # pkl 파일 새로 만들지 결정 (pkl setting 바꿀 때 True)
PATH_SLASH = '/' if MODE == 'linux' else '\\'
IMG_SIZE = 256
VALIDATION_RATIO = 15


### Common ###

# epoch, batch
EPOCHS = 200
SAVING_EPOCH = 2
BATCH_SIZE = 28
BUFFER_SIZE = 3000             # Iterator에 올려놓을 데이터 사이즈(메모리에 올려 놓을 전체 데이터 개수보다 커야 합니다.)

# learning_rate
INIT_LEARNING_RATE = 0.005
DECAY_RATE = 0.9
DECAY_STEP = 2500
DECAY_STAIRCASE = True

# dropout
DROPOUT_RATE = 0.2

# loss
LOSS_FUNC = 'cross_entropy'    # dice, focal, cross_entropy, dice_sum, huber,weighted_cross_entropy

# activation_func

# optimizer

# initialization(weight, bias)

# pooling


### Unet ###

DEPTH = 4                      # Unet 층 깊이 (down * depth + up * depth)
INIT_N_FILTER = 32             # 첫 레이어의 필터수
RGB_CHANNEL = 1                # rgb(3) or greyscale(1)
N_CLASS = 2

### Result ###

PRED_MASKING_COLOR = 'red'     # validation 테스트 예측 영역 마스킹 색 (red or green or blue)
LABEL_MASKING_COLOR = 'blue'   # validation 테스트 라벨 영역 마스킹 색 (red or green or blue)
W = 40                         # 마스킹 비율(값이 낮을 수록 masking 색이 진해집니다.)
P = 0.0001                     # 각 이미지의 적용비율





