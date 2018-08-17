import numpy as np
import os
import utils as utils
from scipy.stats import mode

data_path = ['/home/mspark/project/brats/ensemble/val/1', '/home/mspark/project/brats/ensemble/val/2', '/home/mspark/project/brats/ensemble/val/3']
# data_path = ['d:\\ensemble\\1', 'd:\\ensemble\\2', 'd:\\ensemble\\3']

path_list = []
id_list = os.listdir(data_path[0])

for path in data_path:
    path_list.append([os.path.join(path, p) for p in os.listdir(path)])



for id, i, j, k in zip(id_list, path_list[0], path_list[1], path_list[2]):
    first_npy = np.load(i)
    second_npy = np.load(j)
    third_npy = np.load(k)

    ensemble = first_npy + second_npy + third_npy
    ensemble = np.argmax(ensemble, axis=-1)
    ensemble[ensemble == 3] = 4
    utils.save_array_as_nifty_volume(ensemble, '/home/mspark/project/brats/ensemble/val/soft/{}.nii.gz'.format(id))
    # utils.save_array_as_nifty_volume(ensemble, 'd:\\ensemble\\{}.nii.gz'.format(id[:-4]))




for id, i, j, k in zip(id_list, path_list[0], path_list[1], path_list[2]):
    first_npy = np.argmax(np.load(i), axis=-1).reshape([240,240,155,1])
    second_npy = np.argmax(np.load(j), axis=-1).reshape([240,240,155,1])
    third_npy = np.argmax(np.load(k), axis=-1).reshape([240,240,155,1])

    total_npy = np.concatenate([first_npy, second_npy, third_npy], axis=-1)
    total_npy = mode(total_npy, axis=-1)

    total_npy = np.reshape(total_npy[0], (240,240,155))
    total_npy[total_npy==3] = 4
    # ensemble = first_npy + second_npy + third_npy
    # ensemble = np.argmax(ensemble, axis=-1)
    # ensemble[ensemble == 3] = 4
    utils.save_array_as_nifty_volume(total_npy, '/home/mspark/project/brats/ensemble/val/hard/{}.nii.gz'.format(id[:-4]))
