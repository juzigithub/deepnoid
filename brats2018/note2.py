import nibabel
import numpy as np
from skimage.exposure import rescale_intensity
import cv2

############## make landmarks of histogram #################
def cal_hm_landmark(arr, max_percent = 99.8, n_divide = 4, standard=False, scale=1):
    if arr.ndim > 1:
        arr = arr.ravel()
    arr_hist_sd, arr_edges_sd = np.histogram(arr, bins = int(np.max(arr) - np.min(arr)))

    hist_mean = int(np.mean(arr))
    black_peak = arr_edges_sd[0] + np.argmax(arr_hist_sd[:hist_mean])
    white_peak = hist_mean + np.argmax(arr_hist_sd[hist_mean:])

    threshold = int((black_peak + white_peak) / 2)
    pc1 = threshold
    pc2 = np.percentile(arr, max_percent)
    ioi = arr[np.where((arr>=pc1) * (arr<=pc2))]
    landmark_list = [np.percentile(ioi, i * (100/n_divide) ) for i in range(n_divide) if not i == 0]
    landmark_list = [pc1] + landmark_list + [pc2]

    if standard:
        std_scale = (scale / pc2)
        landmark_list = [landmark * std_scale for landmark in landmark_list]

    return [int(landmark) for landmark in landmark_list]

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

a_concat = cv2.hconcat([a[0], a[30], a[50], a[80], a[120]]) / np.max(a)
b_concat = cv2.hconcat([b[0], b[30], b[50], b[80], b[120]]) / np.max(b)


############## concat imgs for histogram match #################
c = np.append(a,b, axis=0)

############## make landmarks #################
s = 255
standard_list = cal_hm_landmark(c, n_divide= 10, standard=True, scale=s)
a_list = cal_hm_landmark(a, n_divide=10)
b_list = cal_hm_landmark(b, n_divide=10)

############## print landmarks [pc1, m25, m50, m75, pc2] #################
print('s', standard_list)
print('a',a_list)
print('b',b_list)

a[a<= a_list[0]] = 0
b[b<= b_list[0]] = 0

a_concat1 = cv2.hconcat([a[0], a[30], a[50], a[80], a[120]]) / np.max(a)
b_concat1 = cv2.hconcat([b[0], b[30], b[50], b[80], b[120]]) / np.max(b)

cv2.imshow('aaaa', a_concat1)
cv2.imshow('bbbb', b_concat1)

############## rescale each img based on landmarks #################
a_scaled = hm_rescale(a, a_list, standard_list)
b_scaled = hm_rescale(b, b_list, standard_list)


############## 특정 intenstiy 범위만 이미지 출력하도록 그 외 범위의 값들은 다 지워버리기 #################
a_scaled[a_scaled<7 * s / 10] = 0
b_scaled[b_scaled<7 * s/ 10] = 0
a_scaled[a_scaled>=8 * s/ 10] = 0
b_scaled[b_scaled>=8 * s/ 10] = 0

############## show imgs after histogram match #################
a_scaled_concat = cv2.hconcat([a_scaled[0], a_scaled[30],a_scaled[50],a_scaled[80],a_scaled[120]]) / s
b_scaled_concat = cv2.hconcat([b_scaled[0], b_scaled[30],b_scaled[50],b_scaled[80],b_scaled[120]]) / s

total = cv2.vconcat([a_concat, b_concat, a_scaled_concat, b_scaled_concat])
cv2.imshow('_', total)


cv2.waitKey()