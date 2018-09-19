import cv2
import glob
import os
import config as cfg
# path = 'd:\\ta\\label\\*\\*\\*\\'
# path = '/mnt/sdb/mspark/data/Brain_aneurysm_newest/train/label/*/*/*/'

data_list = glob.glob(cfg.LABEL_PATH + '*.png')
data_list = sorted(data_list)
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
        # w, h = stats[label_idx, cv2.CC_STAT_WIDTH], stats[label_idx, cv2.CC_STAT_HEIGHT]
        x2, y2 = x1 + stats[label_idx, cv2.CC_STAT_WIDTH], y1 + stats[label_idx, cv2.CC_STAT_HEIGHT]


        label_list.append([y1/height, x1/width, y2/height, x2/width])
        cv2.rectangle(img, (x1-5, y1-5), (x2+5, y2+5), (255, 0, 0), 1)

    if label_list != [] :
        with open(path + '/' + fn + '.txt', 'wt') as f:
        # with open(path + '\\' + fn + '.txt', 'wt') as f:

            for label in label_list:
                f.write('{} {} {} {} {}\n'.format(1, str(label[0]), str(label[1]), str(label[2]), str(label[3])))
        cv2.imwrite(path + '/' + 'detection_' + fn + '.png', img)
        # cv2.imwrite(path + '\\' + 'detection_' + fn + '.png', img)

        print(path + '/' + fn + ' saved')
        # print(path + '\\' + fn + ' saved')