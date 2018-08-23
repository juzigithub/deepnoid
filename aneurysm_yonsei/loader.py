import os
import numpy as np
import cv2
import config as cfg
import utils

def get_file_path_list(data_path):
   x_path_list = []
   y_path_list = []
   for root, dirs, files in os.walk(data_path, topdown=False):
       for name in files:
           if root[-len('/x') : ] == '/x':
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
    x_path_list, y_path_list = get_file_path_list(data_path)
    npy_list = []


    for x, y in zip(x_path_list, y_path_list):
        x_img = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
        x_img = cv2.resize(x_img, (cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]), interpolation=cv2.INTER_AREA)

        x_img = (x_img - np.mean(x_img)) / np.max(x_img)
        x_img = np.expand_dims(x_img, axis=0)

        y_img = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        y_img = cv2.resize(y_img, (cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]), interpolation=cv2.INTER_AREA)


        x_img = utils.extract_patches_from_batch(x_img, (cfg.PATCH_SIZE, cfg.PATCH_SIZE), cfg.PATCH_STRIDE)
        y_img = utils.extract_patches_from_batch(y_img, (cfg.PATCH_SIZE, cfg.PATCH_SIZE), cfg.PATCH_STRIDE)


        y_img_fg = cv2.threshold(y_img, 50, 1, cv2.THRESH_BINARY)[1]
        y_img_fg = y_img_fg.reshape((1, cfg.PATCH_SIZE, cfg.PATCH_SIZE, 1))
        y_img_bg = cv2.threshold(y_img, 50, 1, cv2.THRESH_BINARY_INV)[1]
        y_img_bg = y_img_bg.reshape((1, cfg.PATCH_SIZE, cfg.PATCH_SIZE, 1))

        y_img = np.concatenate((y_img_bg, y_img_fg), axis=-1)
        y_img = np.argmax(y_img, axis=-1)

        x_y_img = np.concatenate((x_img, y_img), axis=0)

        npy_list.append(x_y_img)
    np.save(save_path + filename, npy_list)


# if __name__ == '__main__':
#     data_path = 'C:\\Users\\sunki\\PycharmProjects\\deepnoid\\aneurysm_yonsei'
#     save_resized_dcm_as_npy(data_path, '.\\', 'sample.npy')
#     sample = np.load('.\\sample.npy')
