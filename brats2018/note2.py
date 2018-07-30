import nibabel
import numpy as np
from skimage.exposure import rescale_intensity
import cv2

############## make landmarks of histogram #################
def cal_hm_landmark(arr, max_percent = 99.8, standard=False, scale=1):
    if arr.ndim > 1:
        arr = arr.ravel()
    arr_hist_sd, arr_edges_sd = np.histogram(arr, bins = int(np.max(arr) - np.min(arr)))
    hist_mean = int(np.mean(arr))
    black_peak = arr_edges_sd[0] + np.argmax(arr_hist_sd[:hist_mean])
    white_peak = arr_edges_sd[0] + hist_mean + np.argmax(arr_hist_sd[hist_mean:])

    threshold = int((black_peak + white_peak) / 2)
    pc1 = threshold
    pc2 = np.percentile(arr, max_percent)
    ioi = arr[np.where((arr>=pc1) * (arr<=pc2))]
    m25 = np.percentile(ioi, 25)
    m50 = np.percentile(ioi, 50)
    m75 = np.percentile(ioi, 75)

    if standard:
        std_scale = (scale / pc2)
        pc1 *= std_scale
        pc2 *= std_scale
        m25 *= std_scale
        m50 *= std_scale
        m75 *= std_scale

    return [int(pc1), int(m25), int(m50), int(m75), int(pc2)]

############## scale imgs based on landmarks of histogram #################
def hm_rescale(arr, input_landmark_list, standard_landmark_list):
    arr_shape = arr.shape
    if arr.ndim > 1:
        arr = arr.ravel()
    arr_copy = np.zeros_like(arr)

    scale_idx = np.where((arr < input_landmark_list[0]))

    # 0 ~ pc1 rescale
    arr_copy[scale_idx] = rescale_intensity(arr[scale_idx],
                                            in_range=(input_landmark_list[0] - 1, input_landmark_list[0]),
                                            out_range=(standard_landmark_list[0] - 1, standard_landmark_list[0]))
    # pc1 ~ m25 ~ m50 ~ m75 ~ pc2 rescale
    for idx in range(len(input_landmark_list) - 1):

        scale_idx = np.where((arr >= input_landmark_list[idx]) * (arr < input_landmark_list[idx+1]))
        arr_copy[scale_idx] = rescale_intensity(arr[scale_idx],
                                                in_range=(input_landmark_list[idx], input_landmark_list[idx+1]),
                                                out_range=(standard_landmark_list[idx], standard_landmark_list[idx+1]))
    # pc2 ~ max rescale
    scale_idx = np.where((arr >= input_landmark_list[-1]))
    arr_copy[scale_idx] = rescale_intensity(arr[scale_idx],
                                                in_range=(input_landmark_list[-1], input_landmark_list[-1] + 1),
                                                out_range=(standard_landmark_list[-1], standard_landmark_list[-1] + 1))



    arr_copy = np.clip(arr_copy, a_min=standard_landmark_list[0], a_max=standard_landmark_list[-1])

    return arr_copy.reshape(arr_shape)

############## load nifti img #################
a = nibabel.load('d:\\Brats18_2013_7_1_flair.nii.gz').get_data()
b = nibabel.load('d:\\Brats18_2013_13_1_flair.nii.gz').get_data()

############## 3d shape to 2d shape : (240, 240, 155) -> (155, 240, 240) #################
a = np.transpose(a, (-1,0,1))
b = np.transpose(b, (-1,0,1))

############## for cv2.clahe (uint type으로 해야함) #################
a = a.astype(np.uint16)
b = b.astype(np.uint16)

############## apply CLAHE (value range : 0 ~ 255 * 255) #################
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
for i in range(155):
    b[i] = clahe.apply(a[i]) / 2 + 1000
for i in range(155):
    a[i] = clahe.apply(a[i])
# for i in range(155):
#     b[i] = clahe.apply(b[i])

############## show imgs after clahe #################
cv2.imshow('a1', a[0])
cv2.imshow('a2', a[30])
cv2.imshow('a3', a[50])
cv2.imshow('a4', a[80])
cv2.imshow('a5', a[120])

cv2.imshow('b1', b[0])
cv2.imshow('b2', b[30])
cv2.imshow('b3', b[50])
cv2.imshow('b4', b[80])
cv2.imshow('b5', b[120])

############## concat imgs for histogram match #################
c = np.append(a,b, axis=0)

############## make landmarks #################
s = 255
standard_list = cal_hm_landmark(c, standard=True, scale=s)
a_list = cal_hm_landmark(a)
b_list = cal_hm_landmark(b)

############## print landmarks [pc1, m25, m50, m75, pc2] #################
print('s', cal_hm_landmark(c, standard=True, scale=s))
print('a',cal_hm_landmark(a))
print('b',cal_hm_landmark(b))

############## rescale each img based on landmarks #################
a_scaled = hm_rescale(a, a_list, standard_list)
b_scaled = hm_rescale(b, b_list, standard_list)


############## 특정 intenstiy 범위만 이미지 출력하도록 그 외 범위의 값들은 다 지워버리기 #################
a_scaled[a_scaled<7 * s / 10] = 0
b_scaled[b_scaled<7 * s/ 10] = 0
a_scaled[a_scaled>=8 * s/ 10] = 0
b_scaled[b_scaled>=8 * s/ 10] = 0

############## show imgs after histogram match #################
cv2.imshow('a11', a_scaled[0]/s)
cv2.imshow('a22', a_scaled[30]/s)
cv2.imshow('a33', a_scaled[50]/s)
cv2.imshow('a44', a_scaled[80]/s)
cv2.imshow('a55', a_scaled[120]/s)

cv2.imshow('b11', b_scaled[0]/s)
cv2.imshow('b22', b_scaled[30]/s)
cv2.imshow('b33', b_scaled[50]/s)
cv2.imshow('b44', b_scaled[80]/s)
cv2.imshow('b55', b_scaled[120]/s)

cv2.waitKey()