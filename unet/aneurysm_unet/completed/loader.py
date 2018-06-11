'''
Data Load 관련
by 박무성 주임
'''

"""
데이터 로더입니다.

이미지 사이즈와 백그라운드 라벨 생성 여부를 전달받고 각각의 메소드를 통해 데이터의 절대경로를 만들고 이미지를 불러와서 배치데이터를 만들어줍니다.

numpy와 cv2를 사용하여 구성되어있습니다.


"""
import re
import cv2
import os
import numpy as np
import config as cfg
import file_converter as fc

class DataLoader:
    def __init__(self):
        self.pkl_converter = fc.pkl_converter()
        self.json_converter = fc.json_converter()

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
        x_path = '{0}x'.format(cfg.PATH_SLASH)
        x_pathplus = '{0}x{0}'.format(cfg.PATH_SLASH)
        y_pathplus = '{0}y{0}'.format(cfg.PATH_SLASH)

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
            img = cv2.resize(img, (cfg.IMG_SIZE, cfg.IMG_SIZE), interpolation=cv2.INTER_AREA)

            data.append(img)

        return np.array(data).reshape([-1, cfg.IMG_SIZE, cfg.IMG_SIZE, 1])

    def _read_label_grey_resized(self, data_list):
        if type(data_list) != str:
            data_list = data_list
        elif type(data_list) == str:
            data_list = [data_list]

        data = []
        for file in data_list:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (cfg.IMG_SIZE, cfg.IMG_SIZE), interpolation=cv2.INTER_AREA)
            img1 = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY)[1]
            img2 = cv2.threshold(img, 50, 1, cv2.THRESH_BINARY_INV)[1]
            img1 = img1.reshape([cfg.IMG_SIZE, cfg.IMG_SIZE, 1])
            img2 = img2.reshape([cfg.IMG_SIZE, cfg.IMG_SIZE, 1])
            img = np.concatenate((img1, img2), axis=2)
            data.append(img)

        return np.array(data).reshape([-1, cfg.IMG_SIZE, cfg.IMG_SIZE, 2])


    def _make_address(self, data_list):
        # validation 이미지 저장 시 필요한 파일 정보를 추출합니다.
        address = []
        for data in data_list:
            if 'abnorm' in data:
                add1 = 0
            else :
                add1 = 1
            add2 = int(data.split(os.path.sep)[-4])
            add3 = os.path.basename(data)
            add3 = int(os.path.splitext(add3)[0][4:])
            address.append([add1, add2, add3])
        return address

    def load_data(self, type='pkl',mode='train'):

        if type == 'pkl':
            build = self.pkl_converter._check_pkl()
        elif type == 'json':
            pass


        if build :
            trainX_list, trainY_list = self._data_list_load(cfg.TRAIN_DATA_PATH, mode=mode)
            valX_list, valY_list = self._data_list_load(cfg.VAL_DATA_PATH, mode=mode)
            trainX = self._read_image_grey_resized(trainX_list)
            trainY = self._read_label_grey_resized(trainY_list)

            train_add = self._make_address(trainX_list)
            val_add = self._make_address(valX_list)

            valX = self._read_image_grey_resized(valX_list)
            valY = self._read_label_grey_resized(valY_list)

            total_dataset = [[train_add, trainX], trainY, [val_add, valX], valY]

            if type == 'pkl':
                self.pkl_converter._make_pkl(total_dataset)

            elif type == 'json':
                pass

        return self.pkl_converter.load_pkl()



if __name__ == '__main__':
    loader = DataLoader()
