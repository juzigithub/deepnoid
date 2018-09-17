import glob
import pydicom as dicom
import cv2
import os
import tensorlayer as tl
import numpy as np
import config as cfg


dcm_path = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/input_dcm/*/*/*/'
npz_path = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/npz/'
tl.files.exists_or_mkdir(npz_path)

data_list = glob.glob(dcm_path + '*.dcm')
data_list = sorted(data_list)
# data_list.sort(key=lambda f: int(filter(str.isdigit, f)))

clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
npy_list = []

for data in data_list:
    print(data)
    img = dicom.read_file(data)
    img = img.pixel_array
    img = cv2.resize(img, (cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]), interpolation=cv2.INTER_AREA)
    img = clahe.apply(img)
    img = (img - np.mean(img)) / np.max(img)
    npy_list.append(img)

print(np.shape(npy_list))
np.savez_compressed(os.path.join(npz_path, 'input_original_{}.npz'.format(cfg.IMG_SIZE[0])), all=npy_list)
print('saved')

