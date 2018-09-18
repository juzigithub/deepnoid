import glob
import pydicom as dicom
import cv2
import os
import tensorlayer as tl
import numpy as np
import config as cfg



# dcm_path = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/input_dcm/*/*/*/'
# npz_path = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/npz/'
tl.files.exists_or_mkdir(cfg.NPZ_PATH)

data_list = glob.glob(cfg.DCM_PATH + '*.dcm')
data_list = sorted(data_list)

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

len_npy_list = np.shape(npy_list)[0]
len_chunk = len_npy_list // 5

np.savez_compressed(os.path.join(cfg.NPZ_PATH, 'feature_extractor_pretrain_input_ori_{}_{}.npz'.format(cfg.IMG_SIZE[0], 0)), chunk0=npy_list[len_chunk*0:len_chunk*1])
np.savez_compressed(os.path.join(cfg.NPZ_PATH, 'feature_extractor_pretrain_input_ori_{}_{}.npz'.format(cfg.IMG_SIZE[0], 1)), chunk1=npy_list[len_chunk*1:len_chunk*2])
np.savez_compressed(os.path.join(cfg.NPZ_PATH, 'feature_extractor_pretrain_input_ori_{}_{}.npz'.format(cfg.IMG_SIZE[0], 2)), chunk2=npy_list[len_chunk*2:len_chunk*3])
np.savez_compressed(os.path.join(cfg.NPZ_PATH, 'feature_extractor_pretrain_input_ori_{}_{}.npz'.format(cfg.IMG_SIZE[0], 3)), chunk3=npy_list[len_chunk*3:len_chunk*4])
np.savez_compressed(os.path.join(cfg.NPZ_PATH, 'feature_extractor_pretrain_input_ori_{}_{}.npz'.format(cfg.IMG_SIZE[0], 4)), chunk4=npy_list[len_chunk*4: ])


# np.save(os.path.join(cfg.NPZ_PATH, 'feature_extractor_pretrain_input_ori_{}_{}.npy'.format(cfg.IMG_SIZE[0], 0)), npy_list[len_chunk*0:len_chunk*1])
# np.save(os.path.join(cfg.NPZ_PATH, 'feature_extractor_pretrain_input_ori_{}_{}.npy'.format(cfg.IMG_SIZE[0], 1)), npy_list[len_chunk*1:len_chunk*2])
# np.save(os.path.join(cfg.NPZ_PATH, 'feature_extractor_pretrain_input_ori_{}_{}.npy'.format(cfg.IMG_SIZE[0], 2)), npy_list[len_chunk*2:len_chunk*3])
# np.save(os.path.join(cfg.NPZ_PATH, 'feature_extractor_pretrain_input_ori_{}_{}.npy'.format(cfg.IMG_SIZE[0], 3)), npy_list[len_chunk*3:len_chunk*4])
# np.save(os.path.join(cfg.NPZ_PATH, 'feature_extractor_pretrain_input_ori_{}_{}.npy'.format(cfg.IMG_SIZE[0], 4)), npy_list[len_chunk*4: ])


print('saved')

