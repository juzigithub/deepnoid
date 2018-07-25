import nibabel
import tensorlayer as tl
import os
import numpy as np
from sklearn.preprocessing import scale
from sklearn.feature_extraction import image


USED_MODALITY = ['flair', 't1', 't1ce', 't2']
INPUT_CHANNEL = len(USED_MODALITY)
PATCH_SIZE = 64
PATCH_STRIDE = 16
PATCH_NCR_CUTLINE = 0.01
PATCH_WT_CUTLINE = 0.25

def extract_patches_from_batch(imgs, patch_shape, stride):
    # simple version of sklearn.feature_extraction.image.extract_patches

    # if input imgs are not multiple imgs(just one img), then add axis=0 to make shape like [batch_size, w, h, ...]
    if imgs.ndim == 2 or (imgs.ndim == 3 and len(patch_shape) == 3):
        imgs = np.expand_dims(imgs, axis=0)

    patch_shape = (len(imgs),) + patch_shape
    patch_transpose = (3,0,1,2,4,5) if len(patch_shape) == 3 else (4,0,1,2,3,5,6,7)
    patch_reshape = (-1,) + patch_shape[1:]
    patch = image.extract_patches(imgs, patch_shape, extraction_step=stride)

    return patch.transpose(patch_transpose).reshape(patch_reshape)


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
        path_dic = {'flair' : flair_path, 't1' : t1_path, 't1ce' : t1ce_path, 't2' : t2_path}

        if train :
            # cfg.([flair_path, t1_path, t1ce_path, t2_path, seg_path] )
            file_list.append([path_dic[modal] for modal in USED_MODALITY] + [seg_path])
        else :
            # file_list.append([flair_path, t1_path, t1ce_path, t2_path])
            file_list.append([path_dic[modal] for modal in USED_MODALITY])

    return file_list

def get_normalized_img(data_sets, train, task1=True):
    total_list = [[] for _ in range(np.shape(data_sets)[-1])] # [ [flair], [t1], [t1ce], [t2], [seg] ]
    total_norm_list = [[] for _ in range(np.shape(data_sets)[-1])]

    for data in data_sets:
        for idx in range(len(total_list)):
            vol = nibabel.load(data[idx]).get_fdata()
            total_list[idx].append(vol)

    print('np.shape(total_list) : ' , np.shape(total_list)) # (5, 42, 160, 190, 150)
    m, _, h, w, _ = np.shape(total_list)  # m : train 5(flair, t1, t1ce, t2, seg)/ validation or test 4(seg x), h,w : 240(img_size)

    total_list = np.transpose(total_list, [0, 1, 4, 3, 2])
    total_list = np.reshape(total_list, [m, -1, w, h])          # (5, 6300, 190, 160)
    print('np.shape(total_list) : ' , np.shape(total_list))

    if train:
        nonzero_idx = np.where(total_list[INPUT_CHANNEL].sum(axis=(1, 2)) != 0.) if task1 else np.arange(len(total_list[0]), dtype=np.int32)

    for idx, imgset in enumerate(total_list):
        if train:
            imgset = imgset[nonzero_idx]
        # to avoid normalizing seg
        if idx < INPUT_CHANNEL:
            shape = np.shape(imgset)
            imgset = imgset.reshape([len(imgset), -1])
            scale(imgset, axis=1, copy=False)
            imgset = imgset.reshape(shape)
            total_norm_list[idx] = imgset
        elif idx ==INPUT_CHANNEL:
            total_norm_list[idx] = imgset

    X = np.transpose(total_norm_list[0:INPUT_CHANNEL], [1, 2, 3, 0])
    Y = total_norm_list[INPUT_CHANNEL] if train else []

    return X, Y

def data_saver(data_path, save_path, splits, train):
    if train :
        test_sets = nii_names(data_path, train=train)
        print('test_sets_shape', np.shape(test_sets))

        for idx in range(splits):
            # train_sets_X, train_sets_Y = get_normalized_img(train_sets[idx], train=train)
            chunk_X, chunk_Y = get_normalized_img(test_sets[idx], train=train)
            print('idx, chunk_x.shape, chunk_y.shape', idx ,np.shape(chunk_X),np.shape(chunk_Y))

            # def extract_patches_from_batch(imgs, patch_shape, stride):
            # def reconstruct_from_patches_nd(patches, image_shape, stride):
            # def discard_patch_idx(input, cut_line):
            #############
            chunk_X = extract_patches_from_batch(chunk_X, (PATCH_SIZE, PATCH_SIZE, INPUT_CHANNEL), PATCH_STRIDE)
            chunk_Y = extract_patches_from_batch(chunk_Y, (PATCH_SIZE, PATCH_SIZE), PATCH_STRIDE)

            print('chunk_x.shape : ', chunk_X.shape)  # shape :  (n, patch size, patch size, n input channel)
            print('chunk_y.shape : ', chunk_Y.shape)  # shape :  (n, patch size, patch size)

            np.save(save_path + 'brats_image_whole_{}.npy'.format(idx), chunk_X)
            np.save(save_path + 'brats_label_whole_{}.npy'.format(idx), chunk_Y)

            print('whole_patch{}.saved'.format(idx))

            n_ncr = np.count_nonzero(chunk_Y==1, axis=tuple(i for i in range(chunk_Y.ndim) if not i == 0)) / np.prod(chunk_Y.shape[1:])
            n_non_zero = np.count_nonzero(chunk_Y, axis=tuple(i for i in range(chunk_Y.ndim) if not i == 0)) / np.prod(chunk_Y.shape[1:])

            passed_idx = np.where((n_ncr >= PATCH_NCR_CUTLINE) * (n_non_zero >= PATCH_WT_CUTLINE))

            print('passed_chunk_x.shape', chunk_X[passed_idx].shape)
            print('passed_chunk_y.shape', chunk_Y[passed_idx].shape)

            np.save(save_path + 'brats_image_selected_{}.npy'.format(idx), chunk_X[passed_idx])
            np.save(save_path + 'brats_label_selected_{}.npy'.format(idx), chunk_Y[passed_idx])
            #########################
            print('selected_patch{}.saved'.format(idx))
    else :
        test_sets = nii_names(data_path, train=False)
        test_sets_X, _ = get_normalized_img(test_sets, train=train)
        test_sets_X = extract_patches_from_batch(test_sets_X, (PATCH_SIZE, PATCH_SIZE, INPUT_CHANNEL), PATCH_STRIDE)
        print('np.shape(test_sets_X)', np.shape(test_sets_X))
        np.save(save_path + 'brats_val_image.npy', test_sets_X)
        print('saved')


if __name__ == '__main__':
    data_path = '/mnt/sdb/mspark/data/brats2018/MICCAI_BraTS_2018_Data_Training/HGG/'
    save_path = '~~~/npy/train/'
    data_saver(data_path=[data_path], save_path=save_path, splits = INPUT_CHANNEL, train = True)
    # 이러면 환자별로 순서대로 2d 155장 * 패치 225장 -> 155 * 225 장의 패치 나옴