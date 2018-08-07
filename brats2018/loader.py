import tensorlayer as tl
import os
from sklearn.model_selection import KFold
import numpy as np
import nibabel
from sklearn.preprocessing import scale
import csv
import cv2
import config as cfg
import utils
# import brats2018.config as cfg
# import brats2018.utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

def crop_volume_with_bounding_box(volume, min_idx, max_idx):

    output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                           range(min_idx[1], max_idx[1] + 1),
                           range(min_idx[2], max_idx[2] + 1))]

    return output

def get_path_list(data_path):
    id_list = []
    for path in data_path:
        # print('aaaaaaaaaaaaaaaaaaaa: ', path)
        path_list = tl.files.load_folder_list(path)
        id_list += [os.path.join(path, os.path.basename(p), os.path.basename(p)) for p in path_list]
    return id_list

def nii_names(data_path, train):
    file_list = []  # train [ [flair_path, t1_path, t1ce_path, t2_path, seg_path], ... ]
                    # validation or test [ [flair_path, t1_path, t1ce_path, t2_path],  ... ]

    path_list = get_path_list(data_path)

    for path in path_list:
        flair_path = path + '_flair.nii.gz'
        t1_path = flair_path.replace('flair', 't1')
        t1ce_path = flair_path.replace('flair', 't1ce')
        t2_path = flair_path.replace('flair', 't2')
        seg_path = flair_path.replace('flair', 'seg')
        path_dic = {'flair' : flair_path, 't1' : t1_path, 't1ce' : t1ce_path, 't2' : t2_path}

        if train :
            # file_list.append([flair_path, t1_path, t1ce_path, t2_path, seg_path] )
            file_list.append([path_dic[modal] for modal in cfg.USED_MODALITY] + [seg_path])
        else :
            # file_list.append([flair_path, t1_path, t1ce_path, t2_path])
            file_list.append([path_dic[modal] for modal in cfg.USED_MODALITY])

    return file_list

def cv(data_path, splits, shuffle):
    file_list = nii_names(data_path, train=True)

    kf = KFold(n_splits=splits, shuffle=shuffle, random_state=0)        # fix the seed
    train_idx = []
    test_idx= []
    for train_index, test_index in kf.split(file_list):
        train_idx.append(train_index)
        test_idx.append(test_index)
    train_sets = []
    val_sets = []

    for i in range(splits):
        tr_idx = train_idx[i]
        val_idx = test_idx[i]

        sub_train_set = []
        sub_val_set = []

        for idx in tr_idx:
            train_id = file_list[idx]
            sub_train_set.append(train_id)

        for idx in val_idx:
            val_id = file_list[idx]
            sub_val_set.append(val_id)

        train_sets.append(sub_train_set)
        val_sets.append(sub_val_set)
    return np.array(train_sets), np.array(val_sets)

def get_hm_landmarks(data_sets, n_divide, scale, save_path):
    total_list = [[] for _ in range(np.shape(data_sets)[-1])]

    for data in data_sets:
        for idx in range(len(total_list)):
            vol = nibabel.load(data[idx]).get_data()
            total_list[idx].append(vol)

    m, n, _, _, c = np.shape(total_list)  # m : train 5(flair, t1, t1ce, t2, seg)/ validation or test 4(seg x), h,w : 240(img_size)
    total_hm_std_arr = np.zeros([m, n_divide + 1])
    total_list = np.transpose(total_list, [0, 1, 4, 2, 3])
    total_list = total_list.astype(np.uint16)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    for modal_idx in range(m):
        if modal_idx <= cfg.N_INPUT_CHANNEL - 1 :
            for patient_idx in range(n):
                for img_idx in range(c):
                    total_list[modal_idx][patient_idx][img_idx] = clahe.apply(total_list[modal_idx][patient_idx][img_idx])

    print('clahe finished')

    for modal_idx in range(m):
        if modal_idx <= cfg.N_INPUT_CHANNEL - 1 :
            for patient_idx in range(n):
                total_hm_std_arr[modal_idx] += np.array(utils.cal_hm_landmark(total_list[modal_idx][patient_idx],
                                                                              threshold=cfg.HM_THRESHOLD_TYPE,
                                                                              n_divide=n_divide,
                                                                              standard=True,
                                                                              scale=scale))

    total_hm_std_arr /= n

    np.save(save_path + 'std_landmark.npy', total_hm_std_arr[:cfg.N_INPUT_CHANNEL].astype(int))
    print('landmark saved')


def get_normalized_img(data_sets, train, task1=True):
    total_list = [[] for _ in range(np.shape(data_sets)[-1])] # [ [flair], [t1], [t1ce], [t2], [seg] ]
    total_norm_list = [[] for _ in range(np.shape(data_sets)[-1])]

    modal_dic = {'flair': 0, 't1': 1, 't1ce': 2, 't2': 3}
    used_modal_list = [modal_dic[i] for i in cfg.USED_MODALITY]
    standard_list = np.load(cfg.SAVE_TRAIN_DATA_PATH + 'std_landmark.npy')
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    for data in data_sets:
        for idx in range(len(total_list)):
            vol = nibabel.load(data[idx]).get_data()

            if idx <= cfg.N_INPUT_CHANNEL - 1:
                vol = np.transpose(vol, (-1, 0, 1))
                vol = vol.astype(np.uint16)

                for i in range(155):
                    vol[i] = clahe.apply(vol[i])

                vol_list = utils.cal_hm_landmark(vol, threshold=cfg.HM_THRESHOLD_TYPE, n_divide=cfg.LANDMARK_DIVIDE)
                try:
                    vol = utils.hm_rescale(vol, vol_list, standard_list[idx])
                    # vol = utils.hm_rescale(vol, vol_list, standard_list[used_modal_list[idx]])
                    vol = np.transpose(vol, (1, 2, 0))
                except:
                    print('used_modal_list : {0}, '
                          'standard_list : {1}, idx : {2}'.format(used_modal_list, standard_list, idx))
                    raise Exception

            b_min, b_max = [24, 24, 0] , [215, 215, 154]
            vol = crop_volume_with_bounding_box(vol,b_min,b_max)
            total_list[idx].append(vol)

    print('np.shape(total_list) : ' , np.shape(total_list)) # (5, 42, 160, 192, 150)
    m, _, h, w, _ = np.shape(total_list)  # m : train 5(flair, t1, t1ce, t2, seg)/ validation or test 4(seg x), h,w : 240(img_size)

    total_list = np.transpose(total_list, [0, 1, 4, 3, 2])
    total_list = np.reshape(total_list, [m, -1, w, h])          # (5, 6300, 192, 160)
    print('np.shape(total_list) : ' , np.shape(total_list))

    if train:
        nonzero_idx = np.where(total_list[cfg.N_INPUT_CHANNEL].sum(axis=(1, 2)) != 0.) if task1 else np.arange(len(total_list[0]), dtype=np.int32)

    for idx, imgset in enumerate(total_list):
        if train:
            imgset = imgset[nonzero_idx]
        # to avoid normalizing seg
        if idx < cfg.N_INPUT_CHANNEL:
            shape = np.shape(imgset)
            imgset = imgset.reshape([len(imgset), -1])
            # minmax_scale(imgset, axis=1, copy=False)
            # scale(imgset, axis=1, copy=False)
            # imgset = imgset / (np.max(imgset, axis=1) + 1e-6).reshape([-1,1])
            imgset = imgset.reshape(shape)
            total_norm_list[idx] = imgset
        elif idx == cfg.N_INPUT_CHANNEL:
            total_norm_list[idx] = imgset

    X = np.transpose(total_norm_list[0:cfg.N_INPUT_CHANNEL], [1, 2, 3, 0])  # [n, img_size, img_size, 4(flair, t1, t1ce, t2)]
    Y = total_norm_list[cfg.N_INPUT_CHANNEL] if train else []

    return X, Y

# data_path = HGG_path , save_path = SAVE_SURVIVAL_DATA_PATH
def survival_id_extractor(csv_path, train=True):
    survival_id_list = []
    survival_age_list = []
    survival_survival_list = []
    survival_ResectionStatus_list = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)

        if train:
            for idx, content in enumerate(reader):
                if content[3] == 'GTR':
                    survival_id_list.append(content[0])
                    survival_age_list.append(float(content[1]))
                    survival_survival_list.append(int(content[2]))
                    survival_ResectionStatus_list.append(content[3])
        else:
            for idx, content in enumerate(reader):
                survival_id_list.append(content[0])
                survival_age_list.append(float(content[1]))
                survival_ResectionStatus_list.append(content[2])

    return survival_id_list

def survival_data_saver(data_path, csv_path, save_path, train=True):
    survival_id_list = []
    survival_age_list = []
    survival_survival_list = []
    survival_ResectionStatus_list = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        if train :
            for idx, content in enumerate(reader):
                if content[3] == 'GTR':
                    survival_id_list.append(content[0])
                    survival_age_list.append(float(content[1]))
                    survival_survival_list.append(int(content[2]))
                    survival_ResectionStatus_list.append(content[3])
        else:
            for idx, content in enumerate(reader):
                survival_id_list.append(content[0])
                survival_age_list.append(float(content[1]))
                survival_ResectionStatus_list.append(content[2])

    file_list = []
    survival_path_list = [os.path.join(data_path, os.path.basename(p), os.path.basename(p)) for p in survival_id_list]

    for path in survival_path_list:
        flair_path = path + '_flair.nii.gz'
        t1_path = flair_path.replace('flair', 't1')
        t1ce_path = flair_path.replace('flair', 't1ce')
        t2_path = flair_path.replace('flair', 't2')
        seg_path = flair_path.replace('flair', 'seg')
        path_dic = {'flair' : flair_path, 't1' : t1_path, 't1ce' : t1ce_path, 't2' : t2_path}

        if train :
            # file_list.append([flair_path, t1_path, t1ce_path, t2_path, seg_path] )
            # file_list.append([eval(modal + '_path') for modal in cfg.USED_MODALITY] + [seg_path])
            file_list.append([path_dic[modal] for modal in cfg.USED_MODALITY] + [seg_path])
        else :
            # file_list.append([flair_path, t1_path, t1ce_path, t2_path])
            # file_list.append([eval(modal + '_path') for modal in cfg.USED_MODALITY])
            file_list.append([path_dic[modal] for modal in cfg.USED_MODALITY])

    if train:
        train_sets_X, train_sets_Y= get_normalized_img(file_list, train=train, task1=False)
        print('idx, chunk_x.shape, chunk_y.shape', np.shape(train_sets_X),np.shape(train_sets_Y))

        print('self.chunk_x.shape : ', train_sets_X.shape)  # shape :  (n, 240, 240, 4)
        print('self.chunk_y.shape : ', train_sets_Y.shape)  # shape :  (n, 240, 240, 4)

        np.save(save_path + 'task2_train_image.npy', train_sets_X)
        np.save(save_path + 'task2_train_label.npy', train_sets_Y)
        print('saved')

    else:
        test_sets_X, _ = get_normalized_img(file_list, train=train, task1=False)
        print(np.shape(test_sets_X))
        np.save(save_path + 'task2_val_image.npy', test_sets_X)
        print('saved')

    return survival_id_list

def data_saver(data_path, save_path, splits, train, shuffle=True):
    if cfg.REBUILD_HM_DATA:
        test_sets = nii_names(data_path, train=train)
        get_hm_landmarks(test_sets, n_divide=cfg.LANDMARK_DIVIDE, scale=255, save_path=save_path)
        pass

    if train:
        _, test_sets = cv(data_path, splits, shuffle=shuffle)
        print('test_sets_shape', np.shape(test_sets))

        for idx in range(splits):
            # train_sets_X, train_sets_Y = get_normalized_img(train_sets[idx], train=train)
            chunk_X, chunk_Y = get_normalized_img(test_sets[idx], train=train)
            print('idx, chunk_x.shape, chunk_y.shape', idx ,np.shape(chunk_X),np.shape(chunk_Y))

            # def extract_patches_from_batch(imgs, patch_shape, stride):
            # def reconstruct_from_patches_nd(patches, image_shape, stride):
            # def discard_patch_idx(input, cut_line):
            #############
            chunk_X = utils.extract_patches_from_batch(chunk_X, (cfg.PATCH_SIZE, cfg.PATCH_SIZE, cfg.N_INPUT_CHANNEL), cfg.PATCH_STRIDE)
            chunk_Y = utils.extract_patches_from_batch(chunk_Y, (cfg.PATCH_SIZE, cfg.PATCH_SIZE), cfg.PATCH_STRIDE)

            print('chunk_x.shape : ', chunk_X.shape)  # shape :  (n, patch size, patch size, n input channel)
            print('chunk_y.shape : ', chunk_Y.shape)  # shape :  (n, patch size, patch size)

            np.save(save_path + 'brats_image_whole_{}.npy'.format(idx), chunk_X)
            np.save(save_path + 'brats_label_whole_{}.npy'.format(idx), chunk_Y)

            print('whole_patch{}.saved'.format(idx))

            n_ncr = np.count_nonzero(chunk_Y==1, axis=tuple(i for i in range(chunk_Y.ndim) if not i == 0)) / np.prod(chunk_Y.shape[1:])
            n_non_zero = np.count_nonzero(chunk_Y, axis=tuple(i for i in range(chunk_Y.ndim) if not i == 0)) / np.prod(chunk_Y.shape[1:])

            passed_idx = np.where((n_ncr >= cfg.PATCH_NCR_CUTLINE) * (n_non_zero >= cfg.PATCH_WT_CUTLINE))
            random_idx = np.random.choice(len(chunk_Y), len(passed_idx[0] // 9), replace=False)
            passed_idx = np.unique(np.append(passed_idx, random_idx))

            # passed_idx = utils.discard_patch_idx(chunk_Y, cfg.PATCH_CUTLINE)
            # print('passed', passed_idx)
            print('passed_chunk_x.shape', chunk_X[passed_idx].shape)
            print('passed_chunk_y.shape', chunk_Y[passed_idx].shape)

            np.save(save_path + 'brats_image_selected_{}.npy'.format(idx), chunk_X[passed_idx])
            np.save(save_path + 'brats_label_selected_{}.npy'.format(idx), chunk_Y[passed_idx])
            #########################
            print('selected_patch{}.saved'.format(idx))
    else :
        test_sets = nii_names(data_path, train=False)
        test_sets_X, _ = get_normalized_img(test_sets, train=train)
        test_sets_X = utils.extract_patches_from_batch(test_sets_X, (cfg.PATCH_SIZE, cfg.PATCH_SIZE, cfg.N_INPUT_CHANNEL), cfg.PATCH_STRIDE)
        print('np.shape(test_sets_X)', np.shape(test_sets_X))
        np.save(save_path + 'brats_val_image.npy', test_sets_X)
        print('saved')

# if __name__ == '__main__':
    # data_path = cfg.HGG_DATA_PATH
    # test_sets = nii_names([data_path], train=True)
    # get_hm_landmarks(test_sets, n_divide=10, scale=255, save_path=cfg.SAVE_TRAIN_DATA_PATH, train=False)

