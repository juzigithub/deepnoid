import os
import glob
import pydicom as dicom
import cv2
import numpy as np
from sklearn.preprocessing import scale

INPUT_PATH = 'd:\\ta\\input_dcm_original\\'
TXT_PATH = 'd:\\ta\\label\\'
N_3D_CONTEXT = 9
IMG_SIZE = [256, 256]

txt_dir_list = os.listdir(TXT_PATH)
print(txt_dir_list)
input_data_list = []
label_data_list = []

def load_normalize_dcm(path):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img = dicom.read_file(path)
    img = img.pixel_array
    img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]), interpolation=cv2.INTER_AREA)
    img = clahe.apply(img)
    img = scale(img)
    # img = (img - np.mean(img)) / np.max(img)
    return img

def load_label(path):
    return np.loadtxt(path, delimiter=' ')

for dir in txt_dir_list:
    input_file_num_dic = {}
    input_num_file_dic = {}
    # input_num_path_dic = {}
    input_num_data_dic = {}

    # txt_file_num_dic = {}
    # txt_file_path_dic = {}

    input_sub_dir = INPUT_PATH + dir + '\\*\\*\\*.dcm'
    txt_sub_dir = TXT_PATH + dir + '\\*\\*\\*.txt'
    # print(sub_dir)

    input_path = glob.glob(input_sub_dir)
    input_path = sorted(input_path)

    txt_path = glob.glob(txt_sub_dir)
    txt_path = sorted(txt_path)
    # print(txt_path)
    n_files = len(input_path)
    # print(n_files)

    for idx, i_path in enumerate(input_path):
        i_bn = os.path.basename(i_path)
        i_fn = os.path.splitext(i_bn)[0]
        # print(i_fn)
        input_file_num_dic[i_fn] = idx
        input_num_file_dic[idx] = i_fn
        # input_num_path_dic[idx] = i_path
        input_num_data_dic[idx] = load_normalize_dcm(i_path)

    for idx, t_path in enumerate(txt_path):
        t_bn = os.path.basename(t_path)
        t_fn = os.path.splitext(t_bn)[0]
        # print(t_fn)
        input_data_list.append([input_num_data_dic[max(0, min(n_files-1, (input_file_num_dic[t_fn] - N_3D_CONTEXT//2 + i)))]
                                for i in range(N_3D_CONTEXT)])
        label_data_list.append(load_label(t_path))

    # print(input_path_list)
    # print(label_path_list)

np.save('d:\\input_data.npy', input_data_list)
np.save('d:\\label_data.npy', label_data_list)
print('saved')