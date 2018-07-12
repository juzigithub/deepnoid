import numpy as np
# import cv2
#
# def masking_rgb(img, color=None):
#     if len(np.shape(img)) <= 2:
#         _img = np.expand_dims(img, axis=3)
#     else:
#         _img = img
#     rgb_list = [np.zeros(np.shape(_img)) for _ in range(3)]
#
#     if color == 'yellow':
#         rgb_list[1] = _img
#         rgb_list[2] = _img
#         B, G, R = rgb_list
#
#     elif color != None:
#         rgb_dic = {'blue': 0, 'green': 1, 'red': 2}
#         rgb_list[rgb_dic[color]] = _img
#         B, G, R = rgb_list
#     else:
#         B = G = R = _img
#
#     concat_img = np.concatenate((B, G, R), axis=-1)
#     out_img = concat_img * 255
#
#     return out_img
#
# a = np.load('d:\\sample\\Brats18_CBICA_AXW_1.npy')
# a = np.transpose(a, [-1, 0,1,2])[68]
#
# y = masking_rgb(a[0], 'yellow')
# g = masking_rgb(a[0], 'green')
# r = masking_rgb(a[0], 'red')
# b = masking_rgb( np.full(a[0].shape, 1.),'blue'    )
#
# cv2.imshow('y', y)
# cv2.imshow('g', g)
# cv2.imshow('r', r)
# cv2.imshow('b', b)
#
# cv2.waitKey()
# cv2.destroyAllWindows()

a = [1,2,3]
