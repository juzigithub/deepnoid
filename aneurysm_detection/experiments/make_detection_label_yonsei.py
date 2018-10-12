import cv2
import glob
import os
import numpy as np
# import config as cfg
import nibabel

INPUT_PATH = 'd:\\yonsei\\input'
LABEL_PATH = 'd:\\yonsei\\label'
SAVE_PATH = 'd:\\yonsei\\label'

input_dir_list = os.listdir(INPUT_PATH)
label_dir_list = os.listdir(LABEL_PATH)

print(input_dir_list)
for i_dir, l_dir in zip(input_dir_list, label_dir_list):
    input_path_list = glob.glob(os.path.join(INPUT_PATH, i_dir, '*.dcm'))
    input_path_list = sorted(input_path_list)
    label_path_list = glob.glob(os.path.join(LABEL_PATH, l_dir, '*.gz'))
    label_path_list = sorted(label_path_list)
    # print('in', input_path_list)
    # print('la', label_path_list)

    for label_path in label_path_list:
        labels = nibabel.load(label_path).get_data()
        labels = np.transpose(labels, (2, 0, 1)).astype(np.uint8)
        # print(labels.shape)

        for idx, img in enumerate(labels):
            input_path = input_path_list[idx]
            input_bn = os.path.basename(input_path)
            fn = os.path.splitext(input_bn)[0]
            # print(fn)

            height, width = img.shape
            # print(img.shape)
            # _, binimage = cv2.threshold(label, 0, 1, cv2.THRESH_BINARY)
            # print(binimage)
            _, _, stats, _ = cv2.connectedComponentsWithStats(img)
            label_areas = stats[1:, cv2.CC_STAT_AREA]

            label_list = []
            img = img.copy() * 255 ### why?

            for label_idx in range(len(label_areas)):
                label_idx += 1

                x1, y1 = stats[label_idx, cv2.CC_STAT_LEFT], stats[label_idx, cv2.CC_STAT_TOP]
                x2, y2 = x1 + stats[label_idx, cv2.CC_STAT_WIDTH], y1 + stats[label_idx, cv2.CC_STAT_HEIGHT]
                label_list.append([y1 / height, x1 / width, y2 / height, x2 / width])
                # print(y1-5, x1-5, y2+5, x2+5)

                cv2.rectangle(img, (x1-10, y1-10), (x2+10, y2+10), (255, 0, 0), 1)

            if label_list != [] :
                # print(label_list)
                # with open(path + '/' + fn + '.txt', 'wt') as f:
                with open(os.path.join(SAVE_PATH, l_dir, fn + '.txt'), 'at') as f:
                    for label in label_list:
                        f.write('{} {} {} {} {}\n'.format(1, str(label[0]), str(label[1]), str(label[2]), str(label[3])))
                # cv2.imwrite(path + '/' + 'detection_' + fn + '.png', img)
                cv2.imwrite(os.path.join(SAVE_PATH, l_dir, 'detection' + fn + '.png'), img)


                # print(path + '/' + fn + ' saved')
                print(l_dir + '\\' + fn + ' saved')
            #


# for data in data_list:
#     path = os.path.split(data)[0]
#     # print(path)
#     fn_ext = os.path.split(data)[1]
#     fn = os.path.splitext(fn_ext)[0]
#     # print(fn)
#
#     img = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
#     height, width = img.shape
#     _, binimage = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
#     _, _, stats, _ = cv2.connectedComponentsWithStats(binimage)
#     label_areas = stats[1:, cv2.CC_STAT_AREA]
#
#     label_list = []
#
#     for label_idx in range(len(label_areas)):
#         label_idx += 1
#
#         x1, y1 = stats[label_idx, cv2.CC_STAT_LEFT], stats[label_idx, cv2.CC_STAT_TOP]
#         # w, h = stats[label_idx, cv2.CC_STAT_WIDTH], stats[label_idx, cv2.CC_STAT_HEIGHT]
#         x2, y2 = x1 + stats[label_idx, cv2.CC_STAT_WIDTH], y1 + stats[label_idx, cv2.CC_STAT_HEIGHT]
#
#
#         label_list.append([y1/height, x1/width, y2/height, x2/width])
#         cv2.rectangle(img, (x1-5, y1-5), (x2+5, y2+5), (255, 0, 0), 1)
#
#     if label_list != [] :
#         with open(path + '/' + fn + '.txt', 'wt') as f:
#         # with open(path + '\\' + fn + '.txt', 'wt') as f:
#
#             for label in label_list:
#                 f.write('{} {} {} {} {}\n'.format(1, str(label[0]), str(label[1]), str(label[2]), str(label[3])))
#         cv2.imwrite(path + '/' + 'detection_' + fn + '.png', img)
#         # cv2.imwrite(path + '\\' + 'detection_' + fn + '.png', img)
#
#         print(path + '/' + fn + ' saved')
#         # print(path + '\\' + fn + ' saved')