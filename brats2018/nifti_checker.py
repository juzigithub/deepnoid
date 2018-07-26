import nibabel as nb
import numpy as np
import cv2

def masking_rgb(img, color=None):
    if len(np.shape(img)) <= 2:
        _img = np.expand_dims(img, axis=3)
    else:
        _img = img
    rgb_list = [np.zeros(np.shape(_img)) for _ in range(3)]

    if color == 'yellow':
        rgb_list[1] = _img
        rgb_list[2] = _img
        B, G, R = rgb_list

    elif color != None:
        rgb_dic = {'blue': 0, 'green': 1, 'red': 2}
        rgb_list[rgb_dic[color]] = _img
        B, G, R = rgb_list
    else:
        B = G = R = _img

    concat_img = np.concatenate((B, G, R), axis=-1)
    out_img = concat_img * 255

    return out_img


original_path = 'd:\\Brats18_CBICA_AUE_1_flair.nii.gz'
result_path = 'd:\\Brats18_CBICA_AUE_1.nii.gz'

original_img = nb.load(original_path).get_fdata().transpose((-1,1,0))
original_img /= np.max(original_img)
result_img = nb.load(result_path).get_fdata().transpose((-1,1,0))

print(original_img.shape)
print(result_img.shape)

for i in range(155):
    ncr_mask = masking_rgb(result_img[i], color='green')
    ncr_mask[ncr_mask!=255] = 0
    ed_mask = masking_rgb(result_img[i], color='blue')
    ed_mask[ed_mask!=255*2] = 0
    et_mask = masking_rgb(result_img[i], color='red')
    et_mask[et_mask!=255*4] = 0

    et_tc_wt = ncr_mask + ed_mask + et_mask
    shape = et_tc_wt.shape
    et_tc_wt_mask = et_tc_wt.reshape((-1,3))
    len_mask = len(et_tc_wt_mask)
    et_tc_wt_mask -= (0.9 * et_tc_wt_mask.max(1).reshape([len_mask, -1]) - et_tc_wt_mask.min(1).reshape([len_mask, -1]))
    et_tc_wt_mask = np.clip(et_tc_wt_mask, 0., 1.) * 255
    et_tc_wt_mask = et_tc_wt_mask.reshape(shape)

    ori = masking_rgb(original_img[i])
    result_image = 0.7 * (ori + et_tc_wt_mask)
    print(result_image.shape)
    cv2.imwrite('d:\\img\\result_{}.jpg'.format(i + 1), result_image)
    cv2.imwrite('d:\\img\\original_{}.jpg'.format(i + 1), ori)
