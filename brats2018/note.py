import nibabel
import numpy as np
import pywt
# a = np.array([[[[[1],[5]],
#                 [[3],[4]]]]])
#
# a[a==5] = 2
# print(a)

#
# def pad(array, reference, offset):
#     """
#     array: Array to be padded
#     reference: Reference array with the desired shape
#     offsets: list of offsets (number of elements must be equal to the dimension of the array)
#     """
#     # Create an array of zeros with the reference shape
#     result = np.zeros(reference.shape)
#     # Create a list of slices from offset to offset + shape in each dimension
#     insertHere = [slice(offset[dim], offset[dim] + array.shape[dim]) for dim in range(a.ndim)]
#     # Insert the array in the result at the specified offsets
#     result[insertHere] = a
#     return result
#
# a = np.ones((3,3,3))
# print(np.shape(a))
# b = np.zeros((5,4,3))
# print(np.shape(b))
# offset = [1,0,0]
# c = pad(a, b, offset)
# print(c)
# print(np.shape(c))

# a = np.load('d:\\Brats18_CBICA_AAB_1.npy')
# print(np.shape(a))
# a = np.flip(a,0)
# a = np.flip(a,1)

# a = np.transpose(a, [2, 0,1])
# a = np.transpose(a, [2, 1, 0])
# a = np.transpose(a, [1,0,2])
# a = np.transpose(a, [1,0,2])

# a= np.transpose(a, [2,1,0])
# a=np.transpose(a,[1,2,0] )
# def save_array_as_nifty_volume(data, filename):
#    img = nibabel.Nifti1Image(data, affine=np.eye(4))
#    nibabel.save(img, filename)
#
#
# save_array_as_nifty_volume(a[2], 'd:\\a.nii.gz')
# vol = nibabel.load('d:\\Brats18_2013_12_1_flair.nii.gz').get_fdata()
# vol2 = nibabel.load('d:\\Brats18_2013_12_1_t1.nii.gz').get_fdata()
# vol3 = nibabel.load('d:\\Brats18_CBICA_ASG_1_flair.nii.gz').get_fdata()
# vol4 = nibabel.load('d:\\Brats18_CBICA_ASG_1_seg.nii.gz').get_fdata()
# from sklearn.preprocessing import scale
# import cv2
# all = []
# all.append(vol)
# all.append(vol3)
# all = np.array(all)
# shape = np.shape(all)

# vol = (vol - np.mean(vol)) / np.std(vol)
# vol3 = (vol3 - np.mean(vol3)) / np.std(vol3)
#
#
# print(np.max(vol[:,:,76]))
# print(np.min(vol[:,:,76]))
# print(np.mean(vol[:,:,76]))
# print(np.std(vol[:,:,76]))
#
# print(np.max(vol3[:,:,76]))
# print(np.min(vol3[:,:,76]))
# print(np.mean(vol3[:,:,76]))
# print(np.std(vol3[:,:,76]))
#
# print(np.shape(vol))
# vol = np.transpose(vol,[1,0,2])
# vol3 = np.transpose(vol3, [1,0,2])





# cv2.imshow('a', np.flip(vol[:,:,76], axis=1))
# cv2.imshow('b', np.flip(vol3[:,:,76], axis=1))
# vol4 = np.transpose(vol4 ,[1,0,2]) / 4.0
#
# cv2.imshow('b', np.flip(vol4[:,:,76], axis=1))
#
# vol4[vol4 == 0.] = 2
# vol4[vol4 == 1.] = 0
# vol4[vol4 == 0.5] = 0
# vol4[vol4 == 0.25] = 0
#
#
# cv2.imshow('c', np.flip(vol4[:,:,76], axis=1))
#
#
#
# cv2.waitKey()
# cv2.destroyAllWindows()

#
# all = all.transpose([0,3,1,2])
# all = all.reshape([2, 155, 240, 240])
# all = all.reshape([2, 155, -1])
# # scale(all, axis=1, copy=False)
#
# ball = []
# for idx, a in enumerate(all):
#     scale(a, axis=1, copy=False)
#     a /= (np.max(a, axis=1) + 1e-6).reshape([-1,1])
#     ball.append(a)
#
# ball = np.array(ball)
#     # imgset = imgset / (np.max(imgset, axis=1) + 1e-6).reshape([-1, 1])
# # all = (all- np.mean(all)) /np.std(all)
# all = ball.reshape([2, 155, 240, 240])
#
# all = np.transpose(all,[0,3,2,1])
# a = all[0, :, :,76]
# b = all[1, :, :, 76]
#
#
# print(np.max(a))
# print(np.min(a))
# print(np.mean(a))
# print(np.std(a))
#
# print(np.max(b))
# print(np.min(b))
# print(np.mean(b))
# print(np.std(b))
#
#
# # imgset = imgset.reshape([len(imgset), -1])
# # scale(imgset, axis=1, copy=False)
# # imgset = imgset / (np.max(imgset, axis=1) + 1e-6).reshape([-1, 1])
# # imgset = imgset.reshape(shape)
# # total_list[idx] = imgset
#
#
# print(shape)
#
# # for i in range(240):
# #     for j in range(240):
#         # print(i,j,vol[i][j][78])
#
#
#
#
# # print(np.max(vol))
# # print(np.min(vol))
# # print(np.mean(vol))
# #
# # print(np.max(vol2))
# # print(np.min(vol2))
# # print(np.mean(vol2))
# #
# # print(np.max(vol3))
# # print(np.min(vol3))
# # print(np.mean(vol3))
# #
# # print(np.max(vol4))
# # print(np.min(vol4))
# # print(np.mean(vol4))
# #
# # for i in range(240):
# #     for j in range(240):
# #         print(i,j,vol2[i][j][84])
#
# #
# # print(np.shape(vol))
# # print(vol[105][70][103]) # 회색
# # print(vol[140][105][55]) # 흰색
# # print(vol[86][136][78]) # 검회색
# #
#
#
# a = np.array([[1,2,3,4],
#               [5,6,7,8],
#               [9,10,11,12],
#               [13,14,15,16]])
# w = pywt.Wavelet('haar')
# print(w)
#
# ca ,*_ = pywt.dwt2(a, 'haar')
#
# print(ca)
# dec_hi = w.dec_hi[::-1]
# print(dec_hi)

