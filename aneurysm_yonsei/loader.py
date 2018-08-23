import os
import numpy as np
import cv2
import pydicom as dicom
import config as cfg
import utils as utils

def get_file_path_list(data_path):
   x_path_list = []
   y_path_list = []
   for root, dirs, files in os.walk(data_path, topdown=False):
       for name in files:
           if root[-len('/dcm') : ] == '/dcm':
               # print('x_name : ', name)
               # print('x_root : ', root)
               path = os.path.join(root, name)
               # print('x_path : ', path)
               x_path_list.append(path)
           elif root[-len('/y') : ] == '/y':
               # print('y_name : ', name)
               # print('y_root : ', root)
               path = os.path.join(root, name)
               # print('y_path : ', path)
               y_path_list.append(path)

   return x_path_list, y_path_list



def save_resized_dcm_as_npy(data_path, save_path, filename):

    if cfg.REBUILD_HM_DATA :
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        dcm_path_list, _ = get_file_path_list(data_path)

        n = len(dcm_path_list)

        total_hm_std_arr = np.zeros([n, cfg.LANDMARK_DIVIDE + 1])

        for dcm in dcm_path_list:
            dic = dicom.read_file(dcm)
            dcm_img = dic.pixel_array
            dcm_img = cv2.resize(dcm_img, (cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]), interpolation=cv2.INTER_AREA)
            dcm_img = clahe.apply(dcm_img)
            landmark_list = utils.cal_hm_landmark(dcm_img, threshold=cfg.HM_THRESHOLD_TYPE, n_divide=cfg.LANDMARK_DIVIDE, standard=True)
            total_hm_std_arr += np.array(landmark_list)

        total_hm_std_arr /= n
        print('total_hm_std_arr', total_hm_std_arr)

        np.save(save_path + 'std_landmark.npy', total_hm_std_arr.astype(int))
        print('landmark_saved')


    standard_landmark_list = np.load(save_path + 'std_landmark.npy')
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    x_path_list, y_path_list = get_file_path_list(data_path)
    npy_list = []

    for x, y in zip(x_path_list, y_path_list):
        x_img = dicom.read_file(x)
        x_img = x_img.pixel_array
        x_img = cv2.resize(x_img, (cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]), interpolation=cv2.INTER_AREA)
        x_img = clahe.apply(x_img)
        landmark_list = utils.cal_hm_landmark(x_img, threshold=cfg.HM_THRESHOLD_TYPE, n_divide=cfg.LANDMARK_DIVIDE)

        x_img = utils.hm_rescale(x_img, landmark_list, standard_landmark_list)

        x_img = np.expand_dims(x_img, axis=0)
        y_img = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        y_img = cv2.resize(y_img, (cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]), interpolation=cv2.INTER_AREA)

        x_img = utils.extract_patches_from_batch(x_img, (cfg.PATCH_SIZE, cfg.PATCH_SIZE), cfg.PATCH_STRIDE)

        y_img_fg = cv2.threshold(y_img, 50, 1, cv2.THRESH_BINARY)[1]
        y_img_fg = y_img_fg.reshape((1, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 1))
        y_img_bg = cv2.threshold(y_img, 50, 1, cv2.THRESH_BINARY_INV)[1]
        y_img_bg = y_img_bg.reshape((1, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 1))

        y_img = np.concatenate((y_img_bg, y_img_fg), axis=-1)
        y_img = np.argmax(y_img, axis=-1)

        y_img = utils.extract_patches_from_batch(y_img, (cfg.PATCH_SIZE, cfg.PATCH_SIZE), cfg.PATCH_STRIDE)

        x_img = np.expand_dims(x_img, axis=-1)
        y_img = np.expand_dims(y_img, axis=-1)

        x_y_img = np.concatenate((x_img, y_img), axis=-1)

        npy_list.append(x_y_img)
    np.save(save_path + filename, npy_list)


# if __name__ == '__main__':
#     data_path = 'C:\\Users\\sunki\\PycharmProjects\\deepnoid\\aneurysm_yonsei'
#     save_resized_dcm_as_npy(data_path, '.\\', 'sample.npy')
#     sample = np.load('.\\sample.npy')
