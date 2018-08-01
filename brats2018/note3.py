from sklearn.preprocessing import minmax_scale
import tensorflow as  tf
import cv2
import nibabel
import numpy as np


vol = nibabel.load('d:\\Brats18_2013_12_1_flair.nii.gz').get_fdata()
vol2 = nibabel.load('d:\\Brats18_2013_12_1_t1.nii.gz').get_fdata()
a = np.append(vol[:,:,84], vol2[:,:,84], axis=0)
minmax_a = minmax_scale(a.reshape([2,-1]), axis=1)
minmax_a= np.reshape(minmax_a, [2,240,240,1])
cv2.imshow('a', minmax_a[0])
print(np.shape(minmax_a))

p = 48 # 패치 사이즈
n, h, _, c = np.shape(minmax_a)

# Image to Patches Conversion
pad = [[0,0],[0,0]]
patches = tf.space_to_batch_nd(minmax_a,[p,p],pad)
patches = tf.split(patches,p*p,0)
patches = tf.stack(patches,3)
patches = tf.reshape(patches,[n*(h//p)**2,p,p,c])
# Do processing on patches
# Using patches here to reconstruct
patches_proc = tf.reshape(patches,[n,h//p,h//p,p*p,c])
patches_proc = tf.split(patches_proc,p*p,3)
patches_proc = tf.stack(patches_proc,axis=0)
patches_proc = tf.reshape(patches_proc,[n*p*p,h//p,h//p,c])
reconstructed = tf.batch_to_space_nd(patches_proc,[p, p],pad)

with tf.Session() as sess:
   P,R_n = sess.run([patches, reconstructed])
   print(P.shape)
   print(R_n.shape)

   cv2.imshow('pat0', P[0])
   cv2.imshow('pat10', P[10])
   cv2.imshow('pat20', P[11])
   cv2.imshow('pat30', P[12])
   cv2.imshow('pat40', P[13])
   cv2.imshow('pat50', P[14])
   cv2.imshow('pat60', P[15])
   cv2.imshow('pat70', P[16])
   cv2.imshow('pat80', P[17])
   cv2.imshow('pat90', P[18])
   cv2.imshow('all', R_n[0])

   err = np.sum((R_n-minmax_a)**2)
   print(err)
   cv2.waitKey()
   cv2.destroyAllWindows()