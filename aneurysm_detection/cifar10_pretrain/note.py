import numpy as np
import pydicom as dicom

path1 = 'd:\\FILE00001.dcm'
path2 = 'd:\\ser401img00001.dcm'

dic1 = dicom.read_file(path1)
dcm_img1 = dic1.pixel_array
print(dcm_img1.shape)

dic2 = dicom.read_file(path2)
dcm_img2 = dic2.pixel_array
print(dcm_img2.shape)
