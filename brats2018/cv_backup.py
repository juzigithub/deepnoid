import tensorlayer as tl
import os
from sklearn.model_selection import KFold
import numpy as np

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

    # print(np.array(file_list))
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

if __name__ == '__main__':
    train_sets, test_sets = cv('/DATA_PATH/', splits=5, shuffle=True)
