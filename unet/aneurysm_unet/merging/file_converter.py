import config as cfg
import os
import _pickle as cpickle
import tensorlayer as tl

class json_converter():
    pass


class pkl_converter():
    def __init__(self):
        self.pkl_full_path = cfg.PKL_DATA_PATH + cfg.PATH_SLASH + cfg.PKL_NAME

    def _make_pkl(self, dataset):
        with open(self.pkl_full_path, 'wb') as f:
            cpickle.dump(dataset, f, protocol=3)
            print('Making ' + cfg.PKL_NAME + ' Completed')

    def check_pkl(self, dataset):
        # pkl 저장 폴더 없는 경우 폴더 생성
        # if not os.path.exists(cfg.PKL_DATA_PATH):
        #     os.mkdir(cfg.PKL_DATA_PATH)

        tl.files.exists_or_mkdir(cfg.PKL_DATA_PATH)
        # pkl 파일 없거나 새로 만들어야 할 때(cfg.REBUILD_PKL = TRUE) pkl 파일 생성
        if not os.path.isfile(self.pkl_full_path) or cfg.REBUILD_PKL:
            self._make_pkl(dataset)

    def load_pkl(self):
        with open(self.pkl_full_path, 'rb') as f:
            trainX, trainY, valX, valY = cpickle.load(f)

        return trainX, trainY, valX, valY
    #
    # def _check_pkl(self, mode='train'):
    #     # pkl 저장 폴더 없는 경우 폴더 생성
    #     if not os.path.exists(cfg.PKL_DATA_PATH):
    #         os.mkdir(cfg.PKL_DATA_PATH)
    #     # pkl 파일 없거나 새로 만들어야 할 때(cfg.REBUILD_PKL = TRUE) pkl 파일 생성
    #     if not os.path.isfile(cfg.PKL_DATA_PATH + cfg.PKL_NAME) or cfg.REBUILD_PKL:  #
    #         self._make_pkl(mode=mode)
    #
    #
    # def _make_address(self, data_list):
    #     address = []
    #     for data in data_list:
    #         if 'abnorm' in data:
    #             add1 = 0
    #         else:
    #             add1 = 1
    #         add2 = int(data.split(os.path.sep)[-4])
    #         add3 = os.path.basename(data)
    #         add3 = int(os.path.splitext(add3)[0][4:])
    #         address.append([add1, add2, add3])
    #     return address
    #
    #
    # def _make_pkl(self, mode='train'):
    #     trainX_list, trainY_list = self._data_list_load(cfg.TRAIN_DATA_PATH, mode=mode)
    #     valX_list, valY_list = self._data_list_load(cfg.VAL_DATA_PATH, mode=mode)
    #     trainX = self._read_image_grey_resized(trainX_list)
    #     trainY = self._read_label_grey_resized(trainY_list)
    #
    #     # Validation Acc 계산 시 필요한 abnorm(0) / norm(1) 여부도 저장 -> valX = [abnorm or norm 여부, img data]
    #     train_add = self._make_address(trainX_list)
    #     val_add = self._make_address(valX_list)
    #
    #     valX = self._read_image_grey_resized(valX_list)
    #     valY = self._read_label_grey_resized(valY_list)
    #
    #     total_dataset = [[train_add, trainX], trainY, [val_add, valX], valY]
    #
    #     with open(cfg.PKL_DATA_PATH + cfg.PKL_NAME, 'wb') as f:
    #         cpickle.dump(total_dataset, f, protocol=3)
    #         print('Making ' + cfg.PKL_NAME + ' Completed')
    #
    #
    # def _load_pkl(self, mode='train'):
    #     if mode == 'train':
    #         with open(cfg.PKL_DATA_PATH + cfg.PKL_NAME, 'rb') as f:
    #             trainX, trainY, valX, valY = cpickle.load(f)
    #     else:
    #         pass  ############## mode = 'test' 일 때
    #
    #     return trainX, trainY, valX, valY