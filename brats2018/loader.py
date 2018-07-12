import tensorlayer as tl
import os
from sklearn.model_selection import KFold
import numpy as np
import nibabel
from sklearn.preprocessing import scale
import csv
import config as cfg

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

def crop_volume_with_bounding_box(volume, min_idx, max_idx):

    output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                           range(min_idx[1], max_idx[1] + 1),
                           range(min_idx[2], max_idx[2] + 1))]

    return output

def get_path_list(data_path):
    id_list = []
    for path in data_path:
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

        if train :
            file_list.append([flair_path, t1_path, t1ce_path, t2_path, seg_path] )
        else :
            file_list.append([flair_path, t1_path, t1ce_path, t2_path])
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

def get_normalized_img(data_sets, train):
    total_list = [[] for _ in range(np.shape(data_sets)[-1])] # [ [flair], [t1], [t1ce], [t2], [seg] ]
    for data in data_sets:
        for idx in range(len(total_list)):
            vol = nibabel.load(data[idx]).get_fdata()
            b_min, b_max = [41, 30, 3] , [200, 221, 152]
            vol = crop_volume_with_bounding_box(vol,b_min,b_max)
            total_list[idx].append(vol)

    print('np.shape(total_list) : ' , np.shape(total_list)) # (5, 42, 160, 190, 150)
    m, _, h, w, _ = np.shape(total_list)  # m : train 5(flair, t1, t1ce, t2, seg)/ validation or test 4(seg x), h,w : 240(img_size)

    total_list = np.transpose(total_list, [0, 1, 4, 3, 2])
    total_list = np.reshape(total_list, [m, -1, w, h])          # (5, 6300, 190, 160)
    print('np.shape(total_list) : ' , np.shape(total_list))

    for idx, imgset in enumerate(total_list):
        # to avoid normalizing seg
        if idx < 4:
            shape = np.shape(imgset)
            imgset = imgset.reshape([len(imgset), -1])
            scale(imgset, axis=1, copy=False)
            imgset = imgset / (np.max(imgset, axis=1) + 1e-6).reshape([-1,1])
            imgset = imgset.reshape(shape)
            total_list[idx] = imgset

    X = np.transpose(total_list[0:4], [1, 2, 3, 0])  # [n, img_size, img_size, 4(flair, t1, t1ce, t2)]
    Y = total_list[4] if train else []

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

        if train :
            file_list.append([flair_path, t1_path, t1ce_path, t2_path, seg_path] )
        else :
            file_list.append([flair_path, t1_path, t1ce_path, t2_path])

    if train :
        train_sets_X, train_sets_Y= get_normalized_img(file_list, train=train)
        print('idx, chunk_x.shape, chunk_y.shape', np.shape(train_sets_X),np.shape(train_sets_Y))

        print('self.chunk_x.shape : ', train_sets_X.shape)  # shape :  (n, 240, 240, 4)
        print('self.chunk_y.shape : ', train_sets_Y.shape)  # shape :  (n, 240, 240, 4)

        np.save(save_path + 'task2_train_image.npy', train_sets_X)
        np.save(save_path + 'task2_train_label.npy', train_sets_Y)
        print('saved')

    else :
        test_sets_X, _ = get_normalized_img(file_list, train=train)
        print(np.shape(test_sets_X))
        np.save(save_path + 'task2_val_image.npy', test_sets_X)
        print('saved')

    return survival_id_list

def data_saver(data_path, save_path, splits, train, shuffle=True):
    if train :
        _, test_sets = cv(data_path, splits, shuffle=shuffle)
        print('test_sets_shape', np.shape(test_sets))

        for idx in range(splits):
            # train_sets_X, train_sets_Y = get_normalized_img(train_sets[idx], train=train)
            chunk_X, chunk_Y = get_normalized_img(test_sets[idx], train=train)
            print('idx, chunk_x.shape, chunk_y.shape', idx ,np.shape(chunk_X),np.shape(chunk_Y))

            print('self.chunk_x.shape : ', chunk_X.shape)  # shape :  (n, 240, 240, 4)
            print('self.chunk_y.shape : ', chunk_Y.shape)  # shape :  (n, 240, 240, 4)

            np.save(save_path + 'brats_image_chunk_{}.npy'.format(idx), chunk_X)
            np.save(save_path + 'brats_label_chunk_{}.npy'.format(idx), chunk_Y)
            print('{}.saved'.format(idx))
    else :
        test_sets = nii_names(data_path, train=False)
        test_sets_X, _ = get_normalized_img(test_sets, train=train)
        print('np.shape(test_sets_X)', np.shape(test_sets_X))
        np.save(save_path + 'brats_val_image.npy', test_sets_X)
        print('saved')