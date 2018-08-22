from nipype import Workflow, Node
import os
import pydicom as dicom
import numpy as np
import cv2


def get_file_path_list(data_path):
   dcm_path_list = []
   y_path_list = []
   for root, dirs, files in os.walk(data_path, topdown=False):
       for name in files:
           if root[-len('\dcm') : ] == '\dcm':
               # print('dcm_name : ', name)
               # print('dcm_root : ', root)
               path = os.path.join(root, name)
               # print('dcm_path : ', path)
               dcm_path_list.append(path)
           elif root[-len('\y') : ] == '\y':
               # print('y_name : ', name)
               # print('y_root : ', root)
               path = os.path.join(root, name)
               # print('y_path : ', path)
               y_path_list.append(path)

   return dcm_path_list, y_path_list






def save_resized_dcm_as_npy(data_path, save_path, filename):
    REBUILD_HM_DATA = True
    IMG_SIZE = 256

    if REBUILD_HM_DATA :
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        dcm_path_list, _ = get_file_path_list(data_path)

        to_normalize_list = []

        for dcm in dcm_path_list:
            dic = dicom.read_file(dcm)
            dcm_img = dic.pixel_array
            dcm_img = cv2.resize(dcm_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            dcm_img = clahe.apply(dcm_img)
            to_normalize_list.append(dcm_img)


    dcm_path_list, y_path_list = get_file_path_list(data_path)
    npz_list = []


    for dcm, y in zip(dcm_path_list, y_path_list):
       sub_dict = {}
       dir_name = dcm.split(os.sep)[-4]
       id = os.path.splitext(os.path.basename(dcm))[0]
       sub_dict['id'] = dir_name + '_' + id
       dic = dicom.read_file(dcm)
       dcm_img = dic.pixel_array
       dcm_img = cv2.resize(dcm_img, (256, 256), interpolation=cv2.INTER_AREA)
       # print(dcm_img.shape)
       print(dcm_img.dtype)
       # for i in range(256):
       #     for j in range(256):
       #         if dcm_img[i][j] > 0 :
       #             print(dcm_img[i][j])

       # cv2.imshow('x', dcm_img/np.max(dcm_img))


       sub_dict['x'] = dcm_img

       y_img = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
       y_img = cv2.resize(y_img, (256, 256), interpolation=cv2.INTER_AREA)
       # print(y_img.shape)




       # cv2.imshow('y', y_img)



       sub_dict['y'] = y_img
       npz_list.append(sub_dict)


   #     # cv2.waitKey()
   #
   # np.save(save_path + filename, npz_list)

if __name__ == '__main__':
    data_path = 'C:\\Users\\sunki\\PycharmProjects\\deepnoid\\aneurysm_yonsei'
    save_resized_dcm_as_npy(data_path, '.\\', 'sample.npy')
    sample = np.load('.\\sample.npy')
    print(sample[0]['id'])
