import pydicom as dicom
import cv2
import os
import glob
import numpy as np
#
#
# input_list = [i for i in range(58, 66)]
#
# for idx in input_list:
#     input = dicom.read_file('d:\\ser401img000{}.dcm'.format(idx)).pixel_array
#     input = cv2.resize(input, (256, 256), cv2.INTER_AREA)
#     labels = np.loadtxt('d:\\ser401img000{}.txt'.format(idx))
#     # labels = np.round(label[1:] * 256).astype(np.int8)
#     if labels.ndim == 1:
#         labels = np.expand_dims(labels, axis=0)
#     for label in labels:
#         label = np.round(label[1:] * 256).astype(np.int8)
#         cv2.rectangle(input, (label[1] - 5, label[0] - 5), (label[3] + 5, label[2] + 5), (255, 255, 255), 1)
#     cv2.imwrite('d:\\{}.png'.format(idx), input)


# INPUT_PATH = 'D:\\yonsei\\input\\SEVSH_BA_000002'
# LABEL_PATH = 'D:\\yonsei\\label\\SEVSH_BA_000002_LabelData'
# input = np.load(os.path.join(INPUT_PATH, 'input_resampled.npz'))['all']
# label = np.load(os.path.join(LABEL_PATH, 'label_resampled.npz'))['all']
# print(input.shape)
# print(label.shape)
# a = np.array([[1,2,3],
#               [4,5,6],
#               [7,8,9]])
# print(a)
# # indices = np.array([0, 2])
# # print(a[indices])
# # a = a.reshape((3,-1))
# # a = (a - np.mean(a, axis=-1))
# print(np.std(a, axis=1))
# print(a - np.mean(a, axis=1).reshape(3, -1))


a = np.load('/home/mspark/deepnoid/data/label_3dce.npz')['SEVSH_BA_000002']
print(a.shape)
print(a)
