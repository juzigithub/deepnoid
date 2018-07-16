import nibabel
import numpy as np

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
vol = nibabel.load('d:\\Brats18_2013_12_1_flair.nii.gz').get_fdata()
vol2 = nibabel.load('d:\\Brats18_2013_12_1_t1.nii.gz').get_fdata()
vol3 = nibabel.load('d:\\Brats18_CBICA_ASG_1_flair.nii.gz').get_fdata()
vol4 = nibabel.load('d:\\Brats18_CBICA_ASG_1_t1.nii.gz').get_fdata()

# for i in range(240):
#     for j in range(240):
        # print(i,j,vol[i][j][78])
print(np.max(vol))
print(np.min(vol))
print(np.mean(vol))

print(np.max(vol2))
print(np.min(vol2))
print(np.mean(vol2))

print(np.max(vol3))
print(np.min(vol3))
print(np.mean(vol3))

print(np.max(vol4))
print(np.min(vol4))
print(np.mean(vol4))
#
# for i in range(240):
#     for j in range(240):
#         print(i,j,vol2[i][j][84])

#
# print(np.shape(vol))
# print(vol[105][70][103]) # 회색
# print(vol[140][105][55]) # 흰색
# print(vol[86][136][78]) # 검회색
#


