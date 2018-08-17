import numpy as np
import os
import utils as utils

data_path = ['/home/mspark/project/brats/ensemble/test/1', '/home/mspark/project/brats/ensemble/test/2', '/home/mspark/project/brats/ensemble/test/3']

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
    utils.save_array_as_nifty_volume(ensemble, '/home/mspark/project/brats/ensemble/test/{}.nii.gz'.format(id))

