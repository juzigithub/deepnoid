### Model info ###
# Unet 기본 모델, background loss 반영

### Path info ###

# lINUX #
MODE = 'linux'
DATA_PATH = '/home/mspark/project/data/Brain_Aneurysm_new_dataset'
TRAIN_DATA_PATH = DATA_PATH + '/train'
VAL_DATA_PATH = DATA_PATH + '/test'
PKL_DATA_PATH = DATA_PATH + '/pkl'
PKL_NAME = 'train.pkl'

# WINDOW #
# MODE = 'window'
# DATA_PATH = 'D:\\dataset\\Brain_Aneurysm_new_dataset'
# TRAIN_DATA_PATH = DATA_PATH + '\\train'
# VAL_DATA_PATH = DATA_PATH + '\\test'
# PKL_DATA_PATH = DATA_PATH + '\\pkl'
# PKL_NAME = 'train.pkl'

### file setting ###

DATA_FILE_TYPE = 'pkl'         # 데이터 로드 및 세이브 형식 결정 (pkl or json) json 은 차후 구현
REBUILD_PKL = False            # pkl 파일 새로 만들지 결정 (pkl setting 바꿀 때 True)
PATH_SLASH = '/' if MODE == 'linux' else '\\'
IMG_SIZE = 256
VALIDATION_RATIO = 15

### Common ###

# epoch, batch
EPOCHS = 200
SAVING_EPOCH = 5
BATCH_SIZE = 28
BUFFER_SIZE = 3000             # Iterator에 올려놓을 데이터 사이즈(메모리에 올려 놓을 전체 데이터 개수보다 커야 합니다.)
ACTIVATION_FUNC = 'relu'       # 가능한 활성함수 : relu, lrelu, elu, prelu, selu
LOSS_FUNC = 'dice'             # 가능한 비용함수 : dice, focal, cross_entropy, dice_sum, huber,weighted_cross_entropy
LAMBDA = 0.8                   # foreground loss 반영 비율 : (LAMBDA) / background loss 반영 비율 : (1 - LAMBDA)
OPTIMIZER = 'adam'             # 가능한 옵티마이저 : adam, rmsprop, sgd
INIT_LEARNING_RATE = 0.005     # 시작 시 학습률
DECAY_RATE = 0.9               # 학습률 디케이 비율
DECAY_STEP = 2500
DECAY_STAIRCASE = True
#####################
NORMALIZATION_TYPE = 'batch'   # batch, group
DOWNSAMPLING_TYPE = 'neighbor' # neighbor, maxpool, avgpool
UPSAMPLING_TYPE = 'transpose'  # resize, transpose, add, concat
GROUP_N = 4                    # group_conv, group_norm 의 group 개수
#####################

DROPOUT_RATE = 0.2

### Unet ###

DEPTH = 4                      # Unet 층 깊이 (down * depth + up * depth)
INIT_N_FILTER = 32             # 첫 레이어의 필터수
RGB_CHANNEL = 1                # rgb(3) or greyscale(1)
N_CLASS = 2

### Densenet ###
GROWTH_RATE = 12
THETA = 0.5

### Mobilenet ###
WIDTH_MULTIPLIER = 1.0         # out_channel = in_channel * width_multiplier

### Shufflenet ###
UNIT_N = 2                     # shuffle unit 의 개수
FIRST_DOWNSAMPLING = False     # 첫번째 레이어 다운샘플링 여부
### Result ###

PRED_MASKING_COLOR = 'red'     # validation 테스트 예측 영역 마스킹 색 (red or green or blue)
LABEL_MASKING_COLOR = 'blue'   # validation 테스트 라벨 영역 마스킹 색 (red or green or blue)
W = 40                         # 마스킹 비율(값이 낮을 수록 masking 색이 진해집니다.)
P = 0.0001                     # 각 이미지의 적용비율





