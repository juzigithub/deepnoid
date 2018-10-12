import os
import numpy as np
import cv2
import pydicom as dicom
import tensorlayer as tl
# import config as cfg
import aneurysm_yonsei.utils as utils

def get_file_path_list(data_path):
    tl.files.exists_or_mkdir('.\\nifti')
    x_path_list = []
    for root, dirs, files in os.walk(data_path, topdown=False):
        patient_num = root.split(sep='\\')[-1]
        patient_list = []

        if int(patient_num) == 100 :
            for name in files:
                path = os.path.join(root, name)
                dcm = dicom.read_file(path)
                try :
                    if '3D' in dcm.SeriesDescription:
                        dcm_img = dcm.pixel_array
                        dcm_img = cv2.resize(dcm_img, (768, 768), interpolation=cv2.INTER_AREA)
                        patient_list.append(dcm_img)
                except :
                    if  '3D' in dcm.ProtocolName:
                        print('except')
                        dcm_img = dcm.pixel_array
                        dcm_img = cv2.resize(dcm_img, (768, 768), interpolation=cv2.INTER_AREA)
                        patient_list.append(dcm_img)
            print('--------------', patient_num)
            print(len(patient_list))
            print(np.shape(patient_list))
            nifti_img = np.array(patient_list).transpose((2,1,0))
            utils.save_array_as_nifty_volume(nifti_img, '.\\nifti\\{}.nii.gz'.format(patient_num))
            print('{} saved'.format(patient_num))

       # for name in files:
       #     print('x_name : ', name)
       #     print('x_root : ', root)
       #     path = os.path.join(root, name)
       #     print('x_path : ', path)
       #     x_path_list.append(path)
           # elif root[-len('/y') : ] == '/y':
           #     # print('y_name : ', name)
           #     # print('y_root : ', root)
           #     path = os.path.join(root, name)
           #     # print('y_path : ', path)
           #     y_path_list.append(path)

    return x_path_list#, y_path_list

if __name__ == '__main__':
    data_path = 'D:\\dataset\\Brain_Aneurysm_new_dataset\\full_data\\dcm'
    print(get_file_path_list(data_path))