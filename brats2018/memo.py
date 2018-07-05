import numpy as np

# a = [[[1,2,3,4],
#       [0,1,2,3]],
#      [[1,2,3,4],
#       [3,2,1,0]]]
#
# # print(np.add(a[0][:][0],1))
# # print(np.subtract(a[0][:][1],1))
# # print(np.multiply(a[0][:][2],2))
# # print(np.divide(a[0][:][3],3))
# a = np.array(a, dtype=np.float32)
# print(a[0])
# a[0,:,0] = (a[0,:,0] - np.mean(a[0,:,0]))/2
# print(a[0,:,1])
# print(a[0,:,2])
# print(a[0,:,3])
#
# print(a)

# a = np.array([[[[1,2,3],
#                 [4,5,6]],
#                [[10,11,12],
#                 [13,14,15]]],
#
#               [[[16,17,18],
#                 [19,20,21]],
#                [[22,23,24],
#                 [25,26,27]]],
#
#               [[[28,29,30],
#                 [31,32,33]],
#                [[34,35,36],
#                 [37,38,39]]]])
#
# print(a)
# print(np.shape(a))
# b = np.transpose(a, [0, 3, 1, 2])
# print(b)
# print(np.shape(b))
#
# c = np.reshape(b, [-1, 2, 2])
# print(c)
# print(np.shape(c))


# import nibabel
# import cv2
# a = nibabel.load('D:\\dataset\\someones_epi.nii.gz').get_fdata()
# print(np.shape(a))
# b = np.max(a[26,:,:])
# print(b)
# cv2.imshow('a', a[26,:,:]/b)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

a = 1
for i in [a]:
    print(1)