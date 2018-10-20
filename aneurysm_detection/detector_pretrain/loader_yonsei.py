import os
import glob
import numpy as np
from sklearn.preprocessing import scale, robust_scale


INPUT_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/input'
LABEL_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/label'
SAVE_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/total'
N_3D_CONTEXT = 9

label_dir_list = os.listdir(LABEL_PATH)
label_dir_list = sorted(label_dir_list)
print(label_dir_list)
input_dic = {}
label_dic = {}

for l_dir in label_dir_list:

    inputs = np.load(os.path.join(INPUT_PATH, l_dir, l_dir + '.npz'))[l_dir]
    n_files, height, width = inputs.shape
    inputs = inputs.reshape((n_files, -1))
    inputs = scale(inputs, axis=1)
    inputs = inputs.reshape((n_files, height, width))

    label_dir_path = os.path.join(LABEL_PATH, l_dir)

    label_indicies = np.array([int(os.path.splitext(t)[0]) for t in os.listdir(label_dir_path) if '.txt' in t])
    label_indicies = sorted(label_indicies)
    # print(label_indicies)

    label_paths = glob.glob(os.path.join(label_dir_path, '*.txt'))
    label_paths = sorted(label_paths)
    # print(label_paths)

    input_arr = []
    label_arr = []

    for idx, l_indice in enumerate(label_indicies):
        selected_indices = np.array([max(0, min(n_files - 1, (l_indice - N_3D_CONTEXT // 2 + i))) for i in range(N_3D_CONTEXT)])
        input_arr.append(inputs[selected_indices])
        label_arr.append(np.loadtxt(label_paths[idx], delimiter=' '))

    input_dic[l_dir] = input_arr
    label_dic[l_dir] = label_arr

np.savez_compressed(os.path.join(SAVE_PATH, 'input_3dce_detector.npz'), **input_dic)
np.savez_compressed(os.path.join(SAVE_PATH, 'label_3dce_detector.npz'), **label_dic)



################################################################
# input = np.load(os.path.join(SAVE_PATH, 'input_3dce.npz'))
# label = np.load(os.path.join(SAVE_PATH, 'label_3dce.npz'))

# input_keys = list(input.keys())
# input_keys = sorted(input_keys)
# print('input_keys', input_keys)
# for key in input_keys:
#     print('shape',input[key][0].shape)
#     print('label',label[key][0])