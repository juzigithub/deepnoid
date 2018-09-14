import cv2
import glob
import os
# path = 'D:\\dataset\\Brain_Aneurysm_new_dataset\\full_data\\new_label\\final_filtered_labels\\*\\*\\*\\'
# path = 'd:\\prac\\*\\*\\*\\'
path = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/label/*/*/*/'
data_list = glob.glob(path + '*.png')
# print(data_list)

for data in data_list:
    path = os.path.split(data)[0]
    # print(path)
    fn_ext = os.path.split(data)[1]
    fn = os.path.splitext(fn_ext)[0]
    # print(fn)

    img = cv2.imread(data, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    _, binimage = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binimage)
    label_areas = stats[1:, cv2.CC_STAT_AREA]

    label_list = []

    for label_idx in range(len(label_areas)):
        label_idx += 1

        x1, y1 = stats[label_idx, cv2.CC_STAT_LEFT], stats[label_idx, cv2.CC_STAT_TOP]
        w, h = stats[label_idx, cv2.CC_STAT_WIDTH], stats[label_idx, cv2.CC_STAT_HEIGHT]

        label_list.append([x1/width, y1/height, w/width, h/height])
        cv2.rectangle(img, (x1-5, y1-5), (x1+w+5, y1+h+5), (255, 0, 0), 1)

    if label_list != [] :
        with open(path + fn + '.txt', 'wt') as f:
            for label in label_list:
                f.write('{} {} {} {} {}\n'.format(1, str(label[0]), str(label[1]), str(label[2]), str(label[3])))
        cv2.imwrite(path + 'detection_' + fn + '.png', img)
        print(path + fn + ' saved')