import tensorlayer as tl
import os
import config as cfg
import nibabel
import numpy as np
import cv2
from sklearn.model_selection import KFold
import file_converter as fc


data_types = ['flair', 't1.', 't1ce', 't2']


def get_id_list(data_path):

    path_list = tl.files.load_folder_list(data_path)
    id_list = [os.path.basename(p) for p in path_list]

    return id_list

'''
HGG_path_list = tl.files.load_folder_list(path=cfg.HGG_data_path)
LGG_path_list = tl.files.load_folder_list(path=cfg.LGG_data_path)

HGG_id_list = [os.path.basename(p) for p in HGG_path_list]
LGG_id_list = [os.path.basename(p) for p in LGG_path_list]

'''

HGG_id_list = get_id_list(cfg.HGG_data_path)
LGG_id_list = get_id_list(cfg.LGG_data_path)

def cv(data_path, splits, shuffle = True):

    '''
    :param data_path:
    :param splits: number of splits
    :param shuffle: bool.
    :param X:
    :param Y:
    :return:
    '''

    id_list = get_id_list(data_path)
    kf = KFold(n_splits=splits, shuffle=shuffle, random_state=0)        # fix the seed
    train_idx = []
    test_idx= []
    for train_index, test_index in kf.split(id_list):
        train_idx.append(train_index)
        test_idx.append(test_index)

    train_sets = []
    val_sets = []
    for i in range(splits):
        tr_idx = train_idx[i]
        val_idx = test_idx[i]

        sub_train_set = []
        sub_val_set = []

        for j in range(len(tr_idx)):

            train_id = id_list[tr_idx[j]]
            sub_train_set.append(train_id)

        for j in range(len(val_idx)):
            val_id = id_list[val_idx[j]]
            sub_val_set.append(val_id)

        train_sets.append(sub_train_set)
        val_sets.append(sub_val_set)

    train_sets = np.array(train_sets)
    val_sets = np.array(val_sets)

    return train_sets, val_sets

train_set, val_set = cv(cfg.HGG_data_path, 5)

print('train_set.shape : ' , train_set.shape)
# print('train_set[0] : ' , train_set[0])
# print('train_set[1] : ' , train_set[1])
# print('train_set[2] : ' , train_set[2])
# print('train_set[3] : ' , train_set[3])
# print('train_set[4] : ' , train_set[4])
print('val_set.shape : ' , val_set.shape)
# print('val_set[0] : ' , val_set[0])
# print('val_set[1] : ' , val_set[1])
# print('val_set[2] : ' , val_set[2])
# print('val_set[3] : ' , val_set[3])
# print('val_set[4] : ' , val_set[4])

def nii_names(data_path):

    x_flair = []
    x_t1 = []
    x_t1ce = []
    x_t2 = []
    # y = []

    id_list = get_id_list(data_path)

    # HGG_train 중에 한명 이름(폴더)
    for patient_name in id_list:
        patient_dir = os.path.join(data_path, patient_name)     # ..\\HGG\\Brats18_CBICA_AAB_1
        image_names = os.listdir(patient_dir)                   # ..\\HGG\\Brats18_CBICA_AAB_1\\Brats18_CBICA_AAB_1_flair.nii.gz
                                                                # ..\\HGG\\Brats18_CBICA_AAB_1\\Brats18_CBICA_AAB_1_t1.nii.gz
                                                                # ..\\HGG\\Brats18_CBICA_AAB_1\\Brats18_CBICA_AAB_1_t1ce.nii.gz
                                                                # ..\\HGG\\Brats18_CBICA_AAB_1\\Brats18_CBICA_AAB_1_t2.nii.gz

        # 그안에 nii.gz 4개
        for image_name in image_names:

            if (data_types[0] in image_name):
                x_flair.append(image_name)
            elif (data_types[1] in image_name):
                x_t1.append(image_name)
            elif (data_types[2] in image_name):
                x_t1ce.append(image_name)
            elif (data_types[3] in image_name):
                x_t2.append(image_name)
            # elif (data_types[4] in image_name):
            #     y.append(image_name)

    return x_flair, x_t1, x_t1ce, x_t2

x_flair, x_t1, x_t1ce, x_t2 = nii_names(cfg.HGG_data_path)

'''
def get_mean_sd_v1(data_path,  data_types):

    data_type_mean = []
    data_type_sd = []

    for i in range(len(data_types)):
        id_list = get_id_list(data_path)
        data_list = []
        for j in range(len(id_list)):
            img_path = os.path.join(data_path, id_list[j], nii_names(data_path)[i][j])
            img = nibabel.load(img_path).get_data()
            data_list.append(img)

        data_list = np.asarray(data_list)
        mean = np.mean(data_list)
        sd = np.std(data_list)

        print('mean : ', mean)
        print('sd : ', sd)

        mean.tolist()
        sd.tolist()
        data_type_mean.append(mean)
        data_type_sd.append(sd)
        # np.append(data_type_sd,sd)

        print(data_type_mean)
        print(type(data_type_mean))

    return data_type_mean, data_type_sd

mean, sd = get_mean_sd(cfg.HGG_data_path, data_types)
print(mean)
print(sd)

'''
data_types = ['flair', 't1', 't1ce', 't2']

def get_mean_sd(data_path, id_list, data_types):

    data_type_mean = []
    data_type_sd = []

    for i in range(len(data_types)):

        data_list = []

        for j in id_list:
            img_path = os.path.join(data_path, str(j), str(j) + '_' + data_types[i] + '.nii.gz')
            img = nibabel.load(img_path).get_data()
            data_list.append(img)
            # data_list = np.append(data_list, [img], axis = 0)

        # data_list = np.asarray(data_list)
        print('len(data list) : ', len(data_list))
        mean = np.mean(data_list)                                   # type : np.float64
        sd = np.std(data_list)

        print('mean : ', mean)
        print('sd : ', sd)

        data_type_mean = np.append(data_type_mean, [mean], axis = 0)             # type : np.ndarray
        data_type_sd = np.append(data_type_sd, [sd], axis = 0)

        # print(data_type_mean) # (4,)

        # print('mean.shape : ', data_type_mean.shape)
        # print('sd.shape : ', data_type_sd.shape)

        # mean.tolist()
        # sd.tolist()
        # data_type_mean.append(mean)
        # data_type_sd.append(sd)

        del data_list
    return data_type_mean, data_type_sd

# mean, sd = get_mean_sd(cfg.HGG_data_path, train_set[0], data_types)
# print(mean)
# print(sd)

'''
def get_normalized_img_v1(data_path,data_types):

    X_train = []
    Y_train = []

    id_list = get_id_list(data_path)
    mean, sd = get_mean_sd(data_path, data_types)

    for i in range(len(id_list)):
        all_3d_data = []

        for j in range(len(data_types)):
            img_path = os.path.join(data_path, id_list[i], nii_names(data_path)[j][i])
            img = nibabel.load(img_path).get_data()
            img = (img - mean[j]) / sd[j]
            img = img.astype(np.float32)
            all_3d_data.append(img)

        seg_img_path = os.path.join(data_path, id_list[i], id_list[i] + '_seg.nii.gz')
        seg_img = nibabel.load(seg_img_path).get_data()
        # seg_img = seg_img.transpose(seg_img, (1,0,2))

        for j in range(all_3d_data[0].shape[2]):    # all_3d_data[0] : (240,240,155)
            combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]), axis=2)
            # combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
            # combined_array.astype(np.float32)
            # X_train.append(combined_array)
            # # seg_2d = seg_img[:,:,j]
            # # seg_2d.astype(int)
            # # Y_train.append(seg_2d)
            # Y_train.append(seg_img)


    return combined_array, all_3d_data, seg_img
'''

def get_normalized_img(data_path, id_list, data_types):

    X = np.empty((0,240,240,4))
    Y = np.empty((0,240,240))

    # id_list = get_id_list(data_path)
    mean, sd = get_mean_sd(data_path, id_list, data_types)
    print('mean : ', mean)
    print('sd : ', sd)

    for i in id_list:
        all_3d_data = np.empty((0,240,240,155))             # (4,240,240,155)

        for j in range(len(data_types)):
            img_path = os.path.join(data_path, i, i + '_' + data_types[j] + '.nii.gz')
            img = nibabel.load(img_path).get_data()
            img = (img - mean[j]) / sd[j]                   # (240, 240, 155), numpy.ndarray'
            # img = img.astype(np.float32)
            # all_3d_data.append(img)
            all_3d_data = np.append(all_3d_data,[img],axis=0)

        seg_img_path = os.path.join(data_path, i, i + '_seg.nii.gz')
        seg_img = nibabel.load(seg_img_path).get_data()
        # print('type(seg_img) : ', type(seg_img))    # <class 'numpy.ndarray'>
        # print('seg_img.shape : ', seg_img.shape)    # (240, 240, 155)

        # seg_img = seg_img.transpose(seg_img, (1,0,2))

        for j in range(all_3d_data[0].shape[2]):    # all_3d_data[0] : (240,240,155)
            combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]), axis=2)
            # print('combined_array.shape : ', combined_array.shape) # (240,240,4)
            # print('type(combined_array) : ', type(combined_array.shape)) # <class 'numpy.ndarray'>
            # combined_array = np.transpose(combined_array, (1, 0, 2))#.tolist()
            # X.append(combined_array)    # (155,240,240,4)
            X = np.append(X, [combined_array], axis=0)              # type : numpy.ndarray
            print('X.shape : ', X.shape)
            seg_2d = seg_img[:,:,j]                                 # shape : (240, 240)
            # seg_2d.astype(int)
            # Y.append(seg_2d)
            Y = np.append(Y, [seg_2d], axis=0)                      # type : numpy.ndarray
            print('Y.shape : ', Y.shape)
            # Y.append(seg_img)
        del all_3d_data
    # X = np.asarray(X, dtype=np.float32)
    # Y = np.asarray(Y, dtype=np.float32)
    print('X.shape : ' , X.shape)
    print('Y.shape : ' , Y.shape)
    print('X.type : ', type(X))
    print('Y.type : ', type(Y))
    return X, Y


class DataLoader:

    def __init__(self):

        self.pkl_converter = fc.pkl_converter()
        self.json_converter = fc.json_converter()

        self.data_types = ['flair', 't1', 't1ce', 't2']

        self.splits = 5

        self.train_sets, self.val_sets = cv(cfg.HGG_data_path, self.splits)

        print('self.train_sets.shape : ', self.train_sets.shape)
        print('self.val_sets.shape : ', self.val_sets.shape)

        # self.train_sets_X, self.train_sets_Y, self.val_sets_X, self.val_sets_Y = [], [], [], []
        self.train_sets_X = np.empty((0,26040, 240, 240, 4))
        self.train_sets_Y = np.empty((0,26040, 240, 240))
        self.val_sets_X = np.empty((0,6510,240,240,4))
        self.val_sets_Y = np.empty((0,6510,240,240))
        # self.train_sets_X = np.empty((0,620, 240, 240, 4))
        # self.train_sets_Y = np.empty((0,620, 240, 240))
        # self.val_sets_X = np.empty((0,155,240,240,4))
        # self.val_sets_Y = np.empty((0,155,240,240))

        print('type(self.train_sets_X) : ', type(self.train_sets_X))            # ndarray


        for i in range(self.splits):

            self.train_set_X, self.train_set_Y = get_normalized_img(cfg.HGG_data_path, self.train_sets[i], self.data_types)
            self.val_set_X, self.val_set_Y = get_normalized_img(cfg.HGG_data_path, self.val_sets[i], self.data_types)

            # # --------------------- show data --------------------------------#
            #
            # x = self.train_set_X[10]
            # y = self.train_set_Y[10]
            #
            # print('x.shape : ', x.shape)
            # print('y.shape : ', y.shape)
            #
            # vis_imgs(x, y, 'brats2018/{}/_train_im.png')

            print('type(self.train_set_X) : ', type(self.train_set_X))      # list , self.train_X.shape : (26040, 240, 240, 4)
                                                                            # self.train_Y.shpae : Y.shape :  (26040, 240, 240)

            # self.train_sets_X.append(self.train_set_X)
            # self.train_sets_Y.append(self.train_set_Y)
            # self.val_sets_X.append(self.val_set_X)
            # self.val_sets_Y.append(self.val_set_Y)

            self.train_sets_X = np.append(self.train_sets_X, [self.train_set_X], axis=0)
            self.train_sets_Y = np.append(self.train_sets_Y, [self.train_set_Y], axis=0)
            self.val_sets_X = np.append(self.val_sets_X, [self.val_set_X], axis=0)
            self.val_sets_Y = np.append(self.val_sets_Y, [self.val_set_Y], axis=0)

            # self.train_sets_X[i] = [self.train_set_X]
            # self.train_sets_Y[i] = [self.train_set_Y]
            # self.val_sets_X[i] = [self.val_set_X]
            # self.val_sets_Y[i] = [self.val_set_Y]

            # self.train_sets_X = np.asarray(self.train_sets_X, dtype=np.float32)          # X.shape :  (26040, 240, 240, 4)
        # self.train_sets_Y = np.asarray(self.train_sets_Y, dtype=np.float32)          # Y.shape :  (26040, 240, 240)
        # self.val_sets_X = np.asarray(self.val_sets_X, dtype=np.float32)          # X.shape :  (26040, 240, 240, 4)
        # self.val_sets_Y = np.asarray(self.val_sets_Y, dtype=np.float32)          # Y.shape :  (26040, 240, 240)

        print('type(self.train_sets_X) : ', type(self.train_sets_X.shape))
        print('self.train_sets_X.shape : ', self.train_sets_X.shape)
        print('self.train_sets_Y.shape : ', self.train_sets_Y.shape)
        print('self.val_sets_X.shape : ', self.val_sets_X.shape)
        print('self.val_sets_Y.shape : ', self.val_sets_Y.shape)

        np.save(cfg.HGG_data_path + 'brats_train_image.npy', self.train_sets_X)
        print(' >>>  brats_train_image.npy saved ')
        np.save(cfg.HGG_data_path + 'brats_train_label.npy', self.train_sets_Y)
        print(' >>>  brats_train_label.npy saved ')
        np.save(cfg.HGG_data_path + 'brats_val_image.npy', self.val_sets_X)
        print(' >>>  brats_val_image.npy saved ')
        np.save(cfg.HGG_data_path + 'brats_val_label.npy', self.val_sets_Y)
        print(' >>>  brats_val_label.npy saved ')


    #
    #
    # def load_data(self, type='pkl', mode='train'):
    #
    #     if type == 'pkl':
    #         build = self.pkl_converter._check_pkl()
    #     elif type == 'json':
    #         pass
    #
    #
    #     if build :
    #         trainX = self.train_sets_X
    #         trainY = self.train_sets_Y
    #         valX = self.val_sets_X
    #         valY = self.val_sets_Y
    #
    #         total_dataset = [trainX, trainY, valX, valY]
    #
    #         if type == 'pkl':
    #             self.pkl_converter._make_pkl(total_dataset)
    #
    #         elif type == 'json':
    #             pass
    #
    #     return self.pkl_converter.load_pkl()
    #
    #
