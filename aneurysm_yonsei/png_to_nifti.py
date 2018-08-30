import cv2
import numpy as np
import os
import tensorlayer as tl
import aneurysm_yonsei.utils as utils


def get_file_path_list(data_path):
    tl.files.exists_or_mkdir('.\\nifti_label')
    x_path_list = []
    for root, dirs, files in os.walk(data_path, topdown=False):
        patient_num = root.split(sep='\\')[-3]
        print(patient_num, type(patient_num))
        patient_list = []
        try:
            int(patient_num)
            for name in files:
                path = os.path.join(root, name)
                png = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                png_img = cv2.resize(png, (768, 768), interpolation=cv2.INTER_AREA)
                png_img_fg = cv2.threshold(png_img, 30, 1, cv2.THRESH_BINARY)[1]
                png_img_fg = png_img_fg.reshape((768, 768, 1))
                png_img_bg = cv2.threshold(png_img, 30, 1, cv2.THRESH_BINARY_INV)[1]
                png_img_bg = png_img_bg.reshape((768, 768, 1))

                png_img = np.concatenate((png_img_bg, png_img_fg), axis=-1)
                png_img = np.argmax(png_img, axis=-1)

                patient_list.append(png_img)

            print('--------------', patient_num)
            print(len(patient_list))
            print(np.shape(patient_list))
            nifti_img = np.array(patient_list).transpose((2,1,0))
            utils.save_array_as_nifty_volume(nifti_img, '.\\nifti_label\\{}.nii.gz'.format(patient_num))
            print('{} saved'.format(patient_num))
        except:
            pass
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
    data_path = 'D:\\dataset\\Brain_Aneurysm_new_dataset\\full_data\\new_label\\final_filtered_labels'
    print(get_file_path_list(data_path))