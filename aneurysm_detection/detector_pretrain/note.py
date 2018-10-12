import numpy as np
import pydicom as dicom
import cv2
import os
import tensorflow as tf
# path = 'd:\\a'
#
# a = os.listdir(path)
#
# # path = 'd:\\SEVSH_BA_000026\\ser004img00001.dcm'
# path_l = [os.path.join(path, p) for p in os.listdir(path)]
# clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
#
# for dir in path_l:
#     img = dicom.read_file(dir)
#     img = img.pixel_array.astype(np.uint16)
#     print('ori',img.shape, np.min(img), np.max(img))
#     img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
#     print('resize',img.shape, np.min(img), np.max(img))
#     img = clahe.apply(img)
#     print('clahe',img.shape, np.min(img), np.max(img))
#     # img = scale(img)
#     img = (img - np.mean(img)) / np.max(img)
#     print('z-score',img.shape, np.min(img), np.max(img))
a = np.array([1,2,3])
print(a)