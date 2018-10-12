import os
import glob
import pydicom as dicom
import cv2
import numpy as np
from sklearn.preprocessing import scale
import config as cfg

dcm_dir_list = os.listdir(cfg.TEST_INPUT_PATH)
dcm_dir_list = sorted(dcm_dir_list)
print(dcm_dir_list)
input_data_list = []
label_data_list = []


def load_normalize_dcm(path):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img = dicom.read_file(path)
    img = img.pixel_array
    img = cv2.resize(img, (cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]), interpolation=cv2.INTER_AREA)
    img = clahe.apply(img)
    # img = scale(img)
    img = (img - np.mean(img)) / np.max(img)
    return img

def load_label(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]), interpolation=cv2.INTER_AREA)
    return img

for dir in dcm_dir_list:
    input_file_num_dic = {}
    input_num_file_dic = {}
    input_num_data_dic = {}

    input_sub_dir = cfg.TEST_INPUT_PATH + dir + '/*/*/*.dcm'
    label_sub_dir = cfg.TEST_LABEL_PATH + dir + '/*/*/*.png'

    input_path = glob.glob(input_sub_dir)
    input_path = sorted(input_path)
    label_path = glob.glob(label_sub_dir)
    label_path = sorted(label_path)
    n_files = len(input_path)

    for idx, i_path in enumerate(input_path):
        i_bn = os.path.basename(i_path)
        i_fn = os.path.splitext(i_bn)[0]
        input_file_num_dic[i_fn] = idx
        input_num_file_dic[idx] = i_fn
        input_num_data_dic[idx] = load_normalize_dcm(i_path)

    for idx, l_path in enumerate(label_path):
        l_bn = os.path.basename(l_path)
        l_fn = os.path.splitext(l_bn)[0]
        input_data_list.append([input_num_data_dic[max(0, min(n_files-1 , (input_file_num_dic[l_fn] - cfg.N_3D_CONTEXT//2 + i)))] for i in range(cfg.N_3D_CONTEXT)])
        label_data_list.append(load_label(l_path))
    print('directory : {} / imgs : {} saved'.format(dir, len(label_path)))

np.savez_compressed(cfg.TEST_NPZ_PATH + 'test_input_ori_{}.npz'.format(cfg.IMG_SIZE[0]), all=input_data_list)
np.savez_compressed(cfg.TEST_NPZ_PATH + 'test_label_ori_{}.npz'.format(cfg.IMG_SIZE[0]), all=label_data_list)
print('saved')