# import nibabel
# import numpy as np
# import pywt
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

# from sklearn.preprocessing import minmax_scale
# import tensorflow as  tf
# import cv2


# vol = nibabel.load('d:\\Brats18_2013_12_1_flair.nii.gz').get_fdata()
# vol2 = nibabel.load('d:\\Brats18_2013_12_1_t1.nii.gz').get_fdata()
# a = np.append(vol[:,:,84], vol2[:,:,84], axis=0)
# minmax_a = minmax_scale(a.reshape([2,-1]), axis=1)
# minmax_a= np.reshape(minmax_a, [2,240,240,1])
# cv2.imshow('a', minmax_a[0])
# print(np.shape(minmax_a))
#
# p = 48 # 패치 사이즈
# n, h, _, c = np.shape(minmax_a)
#
# # Image to Patches Conversion
# pad = [[0,0],[0,0]]
# patches = tf.space_to_batch_nd(minmax_a,[p,p],pad)
# patches = tf.split(patches,p*p,0)
# patches = tf.stack(patches,3)
# patches = tf.reshape(patches,[n*(h//p)**2,p,p,c])
# # Do processing on patches
# # Using patches here to reconstruct
# patches_proc = tf.reshape(patches,[n,h//p,h//p,p*p,c])
# patches_proc = tf.split(patches_proc,p*p,3)
# patches_proc = tf.stack(patches_proc,axis=0)
# patches_proc = tf.reshape(patches_proc,[n*p*p,h//p,h//p,c])
# reconstructed = tf.batch_to_space_nd(patches_proc,[p, p],pad)
#
# with tf.Session() as sess:
#     P,R_n = sess.run([patches, reconstructed])
#     print(P.shape)
#     print(R_n.shape)
#
#     cv2.imshow('pat0', P[0])
#     cv2.imshow('pat10', P[10])
#     cv2.imshow('pat20', P[11])
#     cv2.imshow('pat30', P[12])
#     cv2.imshow('pat40', P[13])
#     cv2.imshow('pat50', P[14])
#     cv2.imshow('pat60', P[15])
#     cv2.imshow('pat70', P[16])
#     cv2.imshow('pat80', P[17])
#     cv2.imshow('pat90', P[18])
#     cv2.imshow('all', R_n[0])
#
#     err = np.sum((R_n-minmax_a)**2)
#     print(err)
#     cv2.waitKey()
#     cv2.destroyAllWindows()


# a = ['a','b','c']
# print(a + ['d'])

# a=np.array([0,1,2,0,2])
# idx = np.where(a != 0.)
# print(idx)
# a = a[idx]
# print(a)
# b= np.array([[[0,1,2],
#               [1,2,3],
#               [0,0,0]],
#              [[0,0,0],
#               [0,0,0],
#               [0,0,0]]])
# print(np.shape(b))
# print(b.sum(axis=(1,2)))
# idx2 = np.where(b.sum(axis=(1,2)) != 0. )
# print(idx2)
# b = b[idx2]
# print(b[0].shape)
# c = {'a':1}
# print(c['a'])
# c = {'a':2}
# print(c['a'])
#
#
# import tensorlayer as tl
#
# tl.cost.dice_hard_coe()

# sb.distplot(vol[:,:,84], hist=True, kde=False, rug=True, color='blue')
# for i in range(240):
#     for j in range(240):
#         print(i,j,vol2[i][j][84])
# print(np.max(vol2[:,:,84]))
# print(np.max(vol2[:,:,82]))
# print(np.max(vol2[:,:,70]))
# StandardScaler()

# def minmax_scale(X, feature_range=(0, 1), axis=0, copy=True):
#
#
# a= np.array([[1,2,3,4],
#              [-4,-3,-2,-1],
#              [0,22,300,750],
#              [0,220, 230,570]], dtype=np.float32)
# minmax_scale(a, axis=1, copy=False)
# print(a)





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

# from numpy.lib.stride_tricks import as_strided
# import numbers
#
#
# def make_patches(arr, patch_shape=2, extraction_step=1):
#     arr_ndim = arr.ndim
#
#     if isinstance(patch_shape, numbers.Number):
#         patch_shape = tuple([patch_shape] * arr_ndim)
#     if isinstance(extraction_step, numbers.Number):
#         extraction_step = tuple([extraction_step] * arr_ndim)
#
#     patch_strides = arr.strides
#
#     slices = [slice(None, None, st) for st in extraction_step]
#     indexing_strides = arr[slices].strides
#
#     patch_indices_shape = (np.array(arr.shape) - np.array(patch_shape)) /\
#         np.array(extraction_step) + 1
#
#     shape = tuple(list(patch_indices_shape) + list(patch_shape))
#     strides = tuple(list(indexing_strides) + list(patch_strides))
#
#     patches = as_strided(arr, shape=shape, strides=strides)
#     return patches





#
#
# from itertools import product
# import numpy as np
# from sklearn.feature_extraction import image
#
# def extract_patches_from_batch(imgs, patch_shape, stride):
#     # simple version of sklearn.feature_extraction.image.extract_patches
#
#     # if input imgs are not multiple imgs(just one img), then add axis=0 to make shape like [batch_size, w, h, ...]
#     if imgs.ndim == 2 or (imgs.ndim == 3 and len(patch_shape) == 3):
#         imgs = np.expand_dims(imgs, axis=0)
#
#     patch_shape = (len(imgs),) + patch_shape
#     patch_transpose = (3,0,1,2,4,5) if len(patch_shape) == 3 else (4,0,1,2,3,5,6,7)
#     patch_reshape = (-1,) + patch_shape[1:]
#     patch = image.extract_patches(imgs, patch_shape, extraction_step=stride)
#
#     return patch.transpose(patch_transpose).reshape(patch_reshape)
#
# def reconstruct_from_patches_nd(patches, image_shape, stride):
#     # modified version of sklearn.feature_extraction.image.reconstruct_from_patches_2d
#     i_h, i_w = image_shape[:2]
#     p_h, p_w = patches.shape[1:3]
#     img = np.zeros(image_shape)
#     img_overlapped = np.zeros(image_shape)
#
#     n_h = i_h - p_h + 1
#     n_w = i_w - p_w + 1
#
#     for p, (i, j) in zip(patches, product(range(0,n_h,stride), range(0,n_w,stride))):
#         if patches.ndim == 3:
#             img[i:i + p_h, j:j + p_w] += p
#             img_overlapped[i:i + p_h, j:j + p_w] += 1
#         elif patches.ndim == 4:
#             print(np.shape(img))
#             img[i:i + p_h, j:j + p_w,:] += p
#             img_overlapped[i:i + p_h, j:j + p_w,:] += 1
#     img /= img_overlapped
#
#     return img
#
# img_size = 8
# patch_size = 4
# stride = 2
# out_size = ((img_size - patch_size) // stride) + 1
# n_patch_to_img = (((img_size - patch_size) // stride) + 1) ** 2
#
# a = np.append(np.arange(1,img_size**2 + 1).reshape((1,img_size,img_size)),
#               np.arange(img_size**2 + 1, 2 * (img_size**2) + 1).reshape((1,img_size,img_size)),
#               axis=0)
# a = np.append(a, np.arange( 2 *(img_size ** 2) + 1, 3 * (img_size ** 2) + 1).reshape((1, img_size, img_size)), axis=0)
# # a = np.expand_dims(a, axis=-1)
# print(a)
# print('a_shape', np.shape(a))
# b = extract_patches_from_batch(a, patch_shape=(patch_size, patch_size), stride=stride)
# # print('b_shape', np.shape(b))
# print('\n---------b---------\n', b,'\n---------b---------\n')
# print('\n---------b[0:{}]---------\n'.format(n_patch_to_img), b[0:n_patch_to_img],'\n---------b[0:{}]---------\n'.format(n_patch_to_img))
# c = reconstruct_from_patches_nd(b[0:n_patch_to_img], image_shape=(img_size, img_size), stride=stride)
# print('\n---------c---------\n', c,'\n---------c---------\n')

################################# 새로운 패치 함수 ########################################
# from itertools import product
# import numpy as np
# from sklearn.feature_extraction import image
#
# # patch_size -> patch_shape 으로 변경. 예를 들어 original image shape 이 [n,192,160,4] 인 경우 patch_shape=[64,64,4]를 넣어주면 됨
# def extract_patches_from_batch(imgs, patch_shape, stride):
#     # simple version of sklearn.feature_extraction.image.extract_patches
#
#     # if input imgs are not multiple imgs(just one img), then add axis=0 to make shape like [batch_size, w, h, ...]
#     if imgs.ndim == 2 or (imgs.ndim == 3 and len(patch_shape) == 3):
#         imgs = np.expand_dims(imgs, axis=0)
#
#     patch_shape = (len(imgs),) + patch_shape
#     patch_transpose = (3,0,1,2,4,5) if len(patch_shape) == 3 else (4,0,1,2,3,5,6,7)
#     patch_reshape = (-1,) + patch_shape[1:]
#     patch = image.extract_patches(imgs, patch_shape, extraction_step=stride)
#
#     return patch.transpose(patch_transpose).reshape(patch_reshape)
#
# # image_size -> image_shape 으로 변경. 예를 들어 복원할 original image shape 이 [n,192,160,4] 인 경우 image_shape=[192,160,4]를 넣어주면 됨.
# def reconstruct_from_patches_nd(patches, image_shape, stride):
#     # modified version of sklearn.feature_extraction.image.reconstruct_from_patches_2d
#     i_h, i_w = image_shape[:2]
#     p_h, p_w = patches.shape[1:3]
#     img = np.zeros(image_shape)
#     img_overlapped = np.zeros(image_shape)
#
#     n_h = i_h - p_h + 1
#     n_w = i_w - p_w + 1
#
#     for p, (i, j) in zip(patches, product(range(0,n_h,stride), range(0,n_w,stride))):
#         if patches.ndim == 3:
#             img[i:i + p_h, j:j + p_w] += p
#             img_overlapped[i:i + p_h, j:j + p_w] += 1
#         elif patches.ndim == 4:
#             print(np.shape(img))
#             img[i:i + p_h, j:j + p_w,:] += p
#             img_overlapped[i:i + p_h, j:j + p_w,:] += 1
#     img /= img_overlapped
#
#     return img
# #############################################################
# # 수정 전 #
# def discard_patch(input, cut_line,patch_size,strides):
#    passed = []
#    patches = extract_patches_from_batch(input, patch_size, strides)
#    for i in range(patches.shape[0]):
#        patch = patches[i]
#        indices = np.nonzero(patch)
#        ratio = len(indices[0]) / float((patch.shape[0] * patch.shape[1]))
#        if ratio >= cut_line :
#            passed.append(patch)
#    return passed
# ###########################################################
# # 수정 후 #
# def discard_patch(input, cut_line,patch_size,strides):
#    passed = []
#    patches = extract_patches_from_batch(input, patch_size, strides)
#    for patch in patches: # loop문 사용시 range(patches.shape[0]) 을 사용하는 대신 for patch in patches: 로 사용 가능
#        indices = np.nonzero(patch) # 이 경우 patch = patches[i] 대신 바로 patch 사용 가능
#        ratio = len(indices[0]) / float((patch.shape[0] * patch.shape[1]))
#        if ratio >= cut_line :
#            passed.append(patch)
#    return passed
# ###########################################################
# # 루프문 안 쓴 버전 #
#
# # 커트라인 넘는 인덱스를 리턴하는 함수
# def discard_patch_idx(input, cut_line):
#     # input = patches of label(seg)
#     n_non_zero = np.count_nonzero(input, axis=tuple(i for i in range(input.ndim) if not i == 0)) / np.prod(input.shape[1:])
#     # axis = tuple(i for i in range(input.ndim) if not i == 0) -> axis = 0 을 제외한 나머지 axis 를 모두 출력. 채널(axis=3)이 있는 경우, 없는 경우 모두 포함하도록 ex: (1,2) or (1,2,3)
#     # count_nonzero(input, axis=(1,2)) -> axis=0을 제외하고 0 아닌 횟수 카운트. 각 배치 별로(axis=0) 0 아닌 횟수 출력
#     # np.prod(input.shape[1:]) -> prod 는 곱셈. 또한 shape[1:] 는 채널(axis=3) 이 있는 경우, 없는 경우 모두 포함하도록 사용. 각 배치별 전체 요소 수를 출력. ex: (100, 64, 64) -> 64 * 64 출력
#     # np.where -> 특정 조건을 만족하는 idx 출력. 즉, n_non_zero = non_zero / prod 에서 계산한 전체 중 0 아닌 비율이 >= cut_line 을 넘는 idx만 출력해줌
#     # 그 인덱스를 이용하여 a = patch array (n, 64, 64). 에서 a[idx] 를 하면 전체 중 0 아닌 비율이 cut_line 넘는 패치만 살아남음
#     # ex : a = np.array([[1,2,3],     idx = np.array([0,2]) 인 경우   -> a[idx] = [[1,2,3],
#     #                    [4,5,6],                                                 [7,8,9]] 가 출력됨
#     #                    [7,8,9]])
#     return np.where(n_non_zero >= cut_line)
#
# # 라벨의 패치를 뽑고,
# b = extract_patches_from_batch(a, patch_shape=(patch_size, patch_size), stride=stride)
# # 커트라인 넘는 인덱스 뽑음
# passed_idx = discard_patch_idx(b, 0.5)
# # 인덱스를 이용해 패치를 슬라이싱해주면 끝
# passed_b = b[passed_idx]
# ##########################################################



# ab = np.array([[[0,1,2],
#                 [0,2,3],
#                 [0,0,0]],
#                [[0,1,1],
#                 [0,1,1],
#                 [0,1,1]]])

# print(np.shape(ab))
# print(np.count_nonzero(ab==1, axis=(1,2)))
# print(np.shape(ab))
# print(np.shape(np.argmax(ab, axis=-1)))



# print((ab==1).sum(axis=0))
# print(ab==1)
# print(ab[np.random.randint(len(ab), size=int(0.5 * len(ab)))])

# idx = discard_patch_idx(ab, 0.5)
# print(ab[idx])
#
# import cv2
# ni = np.load('C:\\Users\\sunki\\PycharmProjects\\deepnoid\\brats2018\\brats_label_selected_0.npy')
# ni_ = np.load('C:\\Users\\sunki\\PycharmProjects\\deepnoid\\brats2018\\brats_image_selected_0.npy')
# print(np.shape(ni))
#
# for i in range(10):
#     cv2.imshow('ni{}'.format(i), ni[30+i])
#     cv2.imshow('ni_{}'.format(i), ni_[30 + i]/ np.max(ni_[30 + i]))
#
# print(np.shape(ni[100]))
# cv2.waitKey()
# cv2.destroyAllWindows()

#
import cv2
import numpy as np
# # def augmentation(arr_x, arr_y):
# #
# #
# #
# #     # flip left <-> right
# #     cv2.flip(arr_x, 1)
# #     cv2.flip(arr_y, 1)
#
# a = [[1,2,3],
#      [4,5,6],
#      [7,8,9]]
# a = np.array(a)
# # m = np.float32([[1,0,-1], [0,1,-1]])
# # b = cv2.warpAffine(a, m, (3,3))
# # print(b)
#
# a[2:,2:] += [1]
# print(a)
#
# import skimage.exposure.equalize_adapthist

a= [1,2,3]
a = [0] + a + [4]
print(a)