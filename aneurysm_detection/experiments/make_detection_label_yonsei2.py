import cv2
import os
import numpy as np

LABEL_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/label'
SAVE_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/label'

label_dir_list = os.listdir(LABEL_PATH)
label_dir_list = sorted(label_dir_list)

for l_dir in label_dir_list:
    labels = np.load(os.path.join(LABEL_PATH, l_dir, l_dir + '_LabelData.npz'))['id'].astype(np.uint8)
    for idx, label in enumerate(labels):
        _, _, stats, _ = cv2.connectedComponentsWithStats(label)
        label_areas = stats[1:, cv2.CC_STAT_AREA]
        height, width = label.shape

        label_list = []

        for label_idx in range(len(label_areas)):
            x1, y1 = stats[label_idx + 1, cv2.CC_STAT_LEFT] - 5, stats[label_idx + 1, cv2.CC_STAT_TOP] - 5
            x2, y2 = x1 + stats[label_idx + 1, cv2.CC_STAT_WIDTH] + 5, y1 + stats[label_idx + 1, cv2.CC_STAT_HEIGHT] + 5
            label_list.append([y1/height, x1/width, y2/height, x2/width])

        if label_list != []:
            with open(os.path.join(SAVE_PATH, l_dir, str(idx) + '.txt'), 'wt') as f:
                for l in label_list:
                    f.write('{} {} {} {} {}\n'.format(1, str(l[0]), str(l[1]), str(l[2]), str(l[3])))