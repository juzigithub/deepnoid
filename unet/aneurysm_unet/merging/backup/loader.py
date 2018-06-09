import re
import cv2
import os
import numpy as np
import _pickle as cpickle

# import Unet.aneurysm_unet.merging.config as cfg
import config as cfg

class DataLoader:
    def __init__(self, img_size):
        self.img_size = img_size

    def _try_int(self, ss):
        try:
            return int(ss)
        except:
            return ss

    def _number_key(self, s):
        return [self._try_int(ss) for ss in re.split('([0-9]+)', s)]

    # 파일명 번호 순으로 정렬
    def _sort_by_number(self, files):
        files.sort(key=self._number_key)
        return files

    # 데이터 경로 로더
    def _data_list_load(self, path, mode):


        #################
        x_path = '/x' if cfg.MODE == 'linux' else '\\x'
        y_path = '/y' if cfg.MODE == 'linux' else '\\x'
        x_pathplus = '/x/' if x_path == '/x' else '\\x\\'
        y_pathplus = '/y/' if y_path == '/y' else '\\y\\'
        #################


        if mode == 'train':
            # 데이터셋 경로를 담아 둘 빈 리스트 생성
            image_list = []
            label_list = []
            # 입력된 모든 경로에 대해서 이미지 데이터 경로를 절대경로로 만든 다음 위에서 생성한 리스트에 저장하고 반환
            for data_path in [path]:
                for root, dirs, files in os.walk(data_path):
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)


                        # windows에서는 path가 안 읽힘 : \x나 그런 식으로 바꿔야 될듯함.
                        if x_path in dir_path:
                            if len(os.listdir(dir_path)) != 0:

                                x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]

                                y_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
                                # y_path_list = [path.replace('/x/', '/x_filtered/') for path in y_path_list]
                                y_path_list = [path.replace(x_pathplus, y_pathplus) for path in y_path_list]

                                images_files = self._sort_by_number(x_path_list)
                                labels_files = self._sort_by_number(y_path_list)

                                for image in images_files:
                                    image_list.append(image)
                                    # print('xdata:', image)

                                for label in labels_files:
                                    label_list.append(label)
                                    # print('ydata:', label)

            return image_list, label_list

        elif mode == 'test':
            # 데이터셋 경로를 담아 둘 빈 리스트 생성
            image_list = []
            down_list = []

            # 입력된 모든 경로에 대해서 이미지 데이터 경로를 절대경로로 만든 다음 위에서 생성한 리스트에 저장하고 반환
            for data_path in path:
                for root, dirs, files in os.walk(data_path):
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        if x_path in dir_path:
                            if len(os.listdir(dir_path)) != 0:
                                x_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]

                                images_files = self._sort_by_number(x_path_list)

                                for image in images_files:
                                    image_list.append(image)

            return image_list, len(image_list)


    def _read_image_grey_resized(self, data_list):
        if type(data_list) != str:
            data_list = data_list
        elif type(data_list) == str:
            data_list = [data_list]

        data = []
        for file in data_list:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

            data.append(img)

        return np.array(data).reshape([-1, self.img_size, self.img_size, 1])

    def _read_label_grey_resized(self, data_list):
        if type(data_list) != str:
            data_list = data_list
        elif type(data_list) == str:
            data_list = [data_list]

        data = []
        for file in data_list:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            img1 = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY)[1]
            img2 = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY_INV)[1]
            img1 = img1.reshape([self.img_size, self.img_size, 1])
            img2 = img2.reshape([self.img_size, self.img_size, 1])
            img = np.concatenate((img1, img2), axis=2)
            # print(img)
            data.append(img)

        return np.array(data).reshape([-1, self.img_size, self.img_size, 2])



########################################################################################################
    def _check_pkl(self, mode='train'):
        # pkl 저장 폴더 없는 경우 폴더 생성
        if not os.path.exists(cfg.PKL_DATA_PATH):
            os.mkdir(cfg.PKL_DATA_PATH)
        # pkl 파일 없거나 새로 만들어야 할 때(cfg.REBUILD_PKL = TRUE) pkl 파일 생성
        if not os.path.isfile(cfg.PKL_DATA_PATH + cfg.PKL_NAME) or cfg.REBUILD_PKL: #
            self._make_pkl(mode=mode)

    def _make_address(self, data_list):
        address = []
        for data in data_list:
            if 'abnorm' in data:
                add1 = 0
            else:
                add1 = 1
            add2 = int(data.split(os.path.sep)[-4])
            add3 = os.path.basename(data)
            add3 = int(os.path.splitext(add3)[0][4:])
            address.append([add1, add2, add3])
        return address

    def _make_pkl(self, mode='train'):

        trainX_list, trainY_list = self._data_list_load(cfg.TRAIN_DATA_PATH, mode=mode)
        valX_list, valY_list = self._data_list_load(cfg.VAL_DATA_PATH, mode=mode)
        trainX = self._read_image_grey_resized(trainX_list)
        trainY = self._read_label_grey_resized(trainY_list)



        # Validation Acc 계산 시 필요한 abnorm(0) / norm(1) 여부도 저장 -> valX = [abnorm or norm 여부, img data]
        train_add = self._make_address(trainX_list)
        val_add = self._make_address(valX_list)

        valX = self._read_image_grey_resized(valX_list)
        valY = self._read_label_grey_resized(valY_list)

        total_dataset = [[train_add, trainX], trainY, [val_add, valX], valY]

        with open(cfg.PKL_DATA_PATH + cfg.PKL_NAME, 'wb') as f:
            cpickle.dump(total_dataset, f, protocol=3)
            print('Making ' + cfg.PKL_NAME + ' Completed')

    def _load_pkl(self, mode='train'):
        if mode == 'train' :
            with open(cfg.PKL_DATA_PATH + cfg.PKL_NAME, 'rb') as f:
                trainX, trainY, valX, valY = cpickle.load(f)
        else :
            pass   ############## mode = 'test' 일 때

        return trainX, trainY, valX, valY

    def load_data(self, mode='train'):
        self._check_pkl(mode=mode)

        if mode == 'train' :
            return self._load_pkl(mode=mode)

        else:
            pass    ############## mode = 'test' 일 때

#######################################################################
if __name__ == '__main__':
    loader = DataLoader(cfg.IMG_SIZE)
    # loader._data_list_load(cfg.TRAIN_DATA_PATH, mode='train')
    # a, b = loader._data_list_load(cfg.VAL_DATA_PATH, mode='train')
    # val_batch_xs_list[img_idx].split(os.path.sep)[-5] \
    # + '_' + val_batch_xs_list[img_idx].split(os.path.sep)[-4] \
    # + '_' + val_batch_xs_list[img_idx].split(os.path.sep)[-1]

    # print(a[0].split(os.path.sep)[-5])
    # print(a[0].split(os.path.sep)[-4])
    # print(a[0].split(os.path.sep)[-1])
    a,b,c,d = loader.load_data('train')
    print(c[0])
