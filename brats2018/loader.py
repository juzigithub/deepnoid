import tensorlayer as tl
import os
from sklearn.model_selection import KFold
import numpy as np
import nibabel
import config as cfg

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

def get_path_list(data_path):
    id_list = []
    for path in data_path:
        # print(path)
        path_list = tl.files.load_folder_list(path)
        # print(path_list)
        id_list += [os.path.join(path, os.path.basename(p), os.path.basename(p)) for p in path_list]
    # print(id_list)
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

def get_normalized_img(data_sets, train):
    # flair_list = []
    # t1_list = []
    # t1ce_list = []
    # t2_list = []
    # seg_list = []
    # list_names = [flair_list, t1_list, t1ce_list, t2_list, seg_list]

    total_list = [[] for _ in range(np.shape(data_sets)[-1])] # [ [flair], [t1], [t1ce], [t2], [seg] ]
    for data in data_sets:
        # [list_name.append(nibabel.load(data[idx]).get_fdata()) for idx, list_name in enumerate(list_names)]
        # flair_list.append(nibabel.load(data[0]).get_fdata())
        # t1_list.append(nibabel.load(data[1]).get_fdata())
        # t1ce_list.append(nibabel.load(data[2]).get_fdata())
        # t2_list.append(nibabel.load(data[3]).get_fdata())
        # seg_list.append(nibabel.load(data[4]).get_fdata())

        [total_list[idx].append(nibabel.load(data[idx]).get_fdata()) for idx in range(len(total_list))]

    total_list = [np.array(total_list[idx], dtype=np.float32) for idx in range(len(total_list))]

    ################ Norm ###################
    # total_list = [(total_list[idx] / np.max(total_list[idx])) for idx in range(len(total_list))   ]
    # total_list = [(total_list[idx] - np.mean(total_list[idx])) / np.std(total_list[idx]) if idx <=3 else total_list[idx] for idx in range(len(total_list))]
    total_list[:4] = [(total_list[idx] - np.mean(total_list[idx])) / np.std(total_list[idx]) for idx in range(4)]

    # print(np.shape(total_list))
    m, _, h, w, _ = np.shape(total_list)  # m : train 5(flair, t1, t1ce, t2, seg)/ validation or test 4(seg x), h,w : 240(img_size)

    total_list = np.transpose(total_list, [0, 1, 4, 3, 2])
    total_list = np.reshape(total_list, [m, -1, w, h])
    # print(np.shape(total_list))

    ### show image #############################
    # cv2.imshow('a', total_list[0][1000])
    # cv2.imshow('aa', total_list[4][1000])
    #
    # cv2.imshow('b', total_list[0][150])
    # cv2.imshow('bb', total_list[4][150])

    # cv2.imshow('c', total_list[0][200])
    # cv2.imshow('c1', total_list[1][200])
    # cv2.imshow('c2', total_list[2][200])
    # cv2.imshow('c3', total_list[3][200])
    # cv2.imshow('cc', total_list[4][200])
    #
    # cv2.imshow('d', total_list[0][50])
    # cv2.imshow('dd', total_list[4][50])
    #
    # cv2.imshow('e', total_list[0][600])
    # cv2.imshow('ee', total_list[4][600])

    # cv2.imshow('f', total_list[0][700])
    # cv2.imshow('f1', total_list[1][700])
    # cv2.imshow('f2', total_list[2][700])
    # cv2.imshow('f3', total_list[3][700])
    # cv2.imshow('ff', total_list[4][700])

    # cv2.imshow('g', total_list[0][900])
    # cv2.imshow('gg', total_list[4][900])
    # ni_img = nibabel.Nifti1Image(total_list[0][200], np.eye(4))
    # nibabel.save(ni_img, 'd:\\img.jpg')

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ##################################################
    X = np.reshape(total_list[0:4], [-1, w, h, m-1 if train else m]) # [n, img_size, img_size, 4(flair, t1, t1ce, t2)]
    # Y = np.expand_dims(total_list[4], axis=3) if train else []        # [n, img_size, img_size, 1]
    Y = np.eye(4)[total_list[4]]
    # b = np.eye(4)[a]


    # print('X_shape', np.shape(X), 'Y_shape', np.shape(Y))
    return X, Y     # , seg

def data_saver(data_path, save_path, splits, train):
    if train :
        train_sets, test_sets = cv(data_path, splits, shuffle=True)
        # data_length = len(train_sets[1])*splits + len(test_sets[1])*splits*155
        # chunk_length = data_length // splits
        # print('data_length', data_length)
        # print('chunk_length', chunk_length)

        print('test_sets_shape', np.shape(test_sets))

        # all_X = np.empty((0,2170, 240, 240, 4))
        # all_Y = np.empty((0,2170, 240, 240, 1))
        # all_X = np.empty([1,240,240,4])
        # all_Y = np.empty([1,240,240,1])

        ## 루프문 하나 더 만들어서 [idx][ : ] 로 세분화하기
        for idx in range(splits):
            # train_sets_X, train_sets_Y = get_normalized_img(train_sets[idx], train=train)
            chunk_X, chunk_Y = get_normalized_img(test_sets[idx], train=train)
            print('idx, chunk_x.shape, chunk_y.shape', idx ,np.shape(chunk_X),np.shape(chunk_Y))
            # all_X = np.append(all_X, chunk_X, axis = 0)
            # all_Y = np.append(all_Y, chunk_Y, axis = 0)

            print('self.chunk_x.shape : ', chunk_X.shape)  # shape :  (n, 240, 240, 4)
            print('self.chunk_y.shape : ', chunk_Y.shape)  # shape :  (n, 240, 240, 4)

            np.save(save_path + 'brats_image_chunk_{}.npy'.format(idx), chunk_X)
            np.save(save_path + 'brats_label_chunk_{}.npy'.format(idx), chunk_Y)
            print('{}.saved'.format(idx))

        #
        # np.save(cfg.HGG_data_path + 'brats_image.npy', all_X)
        # print(' >>>  brats_image.npy saved ')
        # np.save(cfg.HGG_data_path + 'brats_train_label.npy', all_Y)
        # print(' >>>  brats_label.npy saved ')



    else :
        train_sets = nii_names(data_path, train=False)
        ##################################################
        # train_data_length = len(train_sets) // splits
        # for idx in range(splits):
        #     train_sets_X, _ = get_normalized_img(train_sets[train_data_length * idx : train_data_length * (idx + 1)], train=train)
        #     print(np.shape(train_sets_X))
        ##################################################
        train_sets_X, _ = get_normalized_img(train_sets, train=train)
        print(np.shape(train_sets_X))
        np.save('./brats_val_image.npy', train_sets_X)
        print('saved')
        # with open('./pickle.pkl', 'wb') as f:
        #     cpickle.dump(train_sets_X, f, protocol=4)
        # print('saved')


if __name__ == '__main__':
    pass
    # path1 = 'D:\\dataset\\BRATS\\2018\\small_data\\HGG\\'
    # path2 = 'D:\\dataset\\BRATS\\2018\\small_data\\LGG\\'
    # path3 = 'D:\\dataset\\BRATS\\2018\\MICCAI_BraTS_2018_Data_Validation\\'
    #
    # path1 = '/home/hshin255/MICCAI_BraTS_2018_Data_Training/HGG/'
    # path2 = '/home/hshin255/MICCAI_BraTS_2018_Data_Training/LGG/'
    # path3 = '/home/mspark/project/data/brats2018/MICCAI_BraTS_2018_Data_Validation/'
    # data_types = ['flair', 't1', 't1ce', 't2']
    # data_saver([path1, path2], 3, True)
    # data_saver([path3],1,False)


    # get_path_list([path1, path2])
    # get_path_list([path3])

    # nii_names([path1, path2], train=True)
    # cv([path1, path2], 3, shuffle=True)

    # id_list = get_id_list(path)
    # print(id_list)

    # train, test = cv(path, 5, shuffle=True, train=True)

    # print(train) # shape = [ n_splits, n_data, 4(flair, t1, t1ce, t2) ]
    # print(np.shape(train))
    # print(test)
    # print(np.shape(test))
    # f, t1, t1c, t2 = nii_names(path)
    # print('f', f)
    # print('t1', t1)
    # print('t1c', t1c)
    # print('t2', t2)

    # get_normalized_img(train[0], train=False)

    # print(np.shape(train[0]))
    # a = nibabel.load(train[0][0][0])
    # print(a.header)
    # print(type(a))
    # b = a.get_fdata()
    # print(b)

