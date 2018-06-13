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

    def _check_pkl(self):
        tl.files.exists_or_mkdir(cfg.PKL_DATA_PATH)

        # pkl 파일 없거나 새로 만들어야 할 때(cfg.REBUILD_PKL = TRUE) pkl 파일 생성
        if not os.path.isfile(self.pkl_full_path) or cfg.REBUILD_PKL:
            return True
        return False

    def load_pkl(self):
        with open(self.pkl_full_path, 'rb') as f:
            trainX, trainY, valX, valY = cpickle.load(f)

        return trainX, trainY, valX, valY
