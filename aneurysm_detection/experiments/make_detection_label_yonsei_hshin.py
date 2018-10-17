import cv2
import os
import numpy as np
import glob

# LABEL_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/label'
# SAVE_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/label'

LABEL_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/label'
INPUT_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/input'
TOTAL_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/total'
SAVE_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/label'

def get_dir_path_list(data_path):
    dir_path_list = []
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in dirs:
            dir_path_list.append(os.path.join(root, name))

    return dir_path_list


def save_bbox_input(data_path, input_path, save_path):
    dir_paths = get_dir_path_list(data_path)
    dict = {}
    for dir in dir_paths:
        id = os.path.basename(dir)
        txt_paths = glob.glob(dir + '\\*.txt')
        z_index = []
        for txt_path in txt_paths:
            base_name = os.path.splitext(os.path.basename(txt_path))[0]
            z_index.append(int(base_name))
        input = np.load(os.path.join(input_path, id, id + '.npz'))
        imgs = input[id]
        img_4d = np.vstack([imgs[np.newaxis, :, :, :] for n in range(len(imgs))])
        new_img = np.stack([img_4d[z, np.clip(range(z - 4, z + 5), a_min=0, a_max=len(img_4d)), :, :] for z in z_index])
        dict[id] = new_img
    np.savez_compressed(os.path.join(save_path, 'input_3dce.npz'), **dict)

label_dir_list = os.listdir(LABEL_PATH)
label_dir_list = sorted(label_dir_list)
dict = {}
for l_dir in label_dir_list:
    labels = np.load(os.path.join(LABEL_PATH, l_dir, l_dir + '_LabelData.npz'))[l_dir].astype(np.uint8)
    all_label = []
    for idx, label in enumerate(labels):
        _, _, stats, _ = cv2.connectedComponentsWithStats(label)
        label_areas = stats[1:, cv2.CC_STAT_AREA]
        height, width = label.shape

        label_list = []
        for label_idx in range(len(label_areas)):
            x1, y1 = stats[label_idx + 1, cv2.CC_STAT_LEFT] - 5, stats[label_idx + 1, cv2.CC_STAT_TOP] - 5
            x2, y2 = x1 + stats[label_idx + 1, cv2.CC_STAT_WIDTH] + 5, y1 + stats[label_idx + 1, cv2.CC_STAT_HEIGHT] + 5
            if [y1 / height, x1 / width, y2 / height, x2 / width] != []:
                label_list.append([1, y1 / height, x1 / width, y2 / height, x2 / width])

        if label_list != []:
            with open(os.path.join(SAVE_PATH, l_dir, str(idx) + '.txt'), 'wt') as f:
                for l in label_list:
                    f.write('{} {} {} {} {}\n'.format(str(l[0]), str(l[1]), str(l[2]), str(l[3]), str(l[4])))

            all_label.append(np.asarray(label_list))

    dict[l_dir] = np.asarray(all_label)


np.savez_compressed(os.path.join(TOTAL_PATH, 'label_3dce'), **dict)
save_bbox_input(LABEL_PATH,INPUT_PATH, TOTAL_PATH)