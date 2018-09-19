import os
import glob
import pydicom as dicom
import cv2
import numpy as np
from sklearn.preprocessing import scale
import config as cfg

txt_dir_list = os.listdir(cfg.TXT_PATH)
txt_dir_list = sorted(txt_dir_list)
print(txt_dir_list)
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
    return np.loadtxt(path, delimiter=' ')

for dir in txt_dir_list:
    input_file_num_dic = {}
    input_num_file_dic = {}
    input_num_data_dic = {}

    input_sub_dir = cfg.INPUT_PATH + dir + '/*/*/*.dcm'
    txt_sub_dir = cfg.TXT_PATH + dir + '/*/*/*.txt'

    input_path = glob.glob(input_sub_dir)
    input_path = sorted(input_path)

    txt_path = glob.glob(txt_sub_dir)
    txt_path = sorted(txt_path)
    n_files = len(input_path)

    for idx, i_path in enumerate(input_path):
        i_bn = os.path.basename(i_path)
        i_fn = os.path.splitext(i_bn)[0]
        input_file_num_dic[i_fn] = idx
        input_num_file_dic[idx] = i_fn
        input_num_data_dic[idx] = load_normalize_dcm(i_path)

    for idx, t_path in enumerate(txt_path):
        t_bn = os.path.basename(t_path)
        t_fn = os.path.splitext(t_bn)[0]
        input_data_list.append([input_num_data_dic[max(0, min(n_files-1, (input_file_num_dic[t_fn] - cfg.N_3D_CONTEXT//2 + i)))]
                                for i in range(cfg.N_3D_CONTEXT)])
        label_data_list.append(load_label(t_path))
    print('directory : {} / imgs : {} saved'.format(dir, len(txt_path)))

np.save(cfg.NPZ_PATH + 'rpn_pretrain_input_ori_{}'.format(cfg.IMG_SIZE[0]), input_data_list)
np.save(cfg.NPZ_PATH + 'rpn_pretrain_label_ori_{}'.format(cfg.IMG_SIZE[0]), label_data_list)
print('saved')
