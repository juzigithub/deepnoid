import os
import glob
import numpy as np
from sklearn.preprocessing import scale

INPUT_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/input'
SAVE_PATH = '/mnt/sdb/mspark/data/brain_aneurysm_yonsei/npz/total'
N_3D_CONTEXT = 9

input_npz_list = glob.glob(os.path.join(INPUT_PATH,'*/*.npz'))
input_npz_list = sorted(input_npz_list)
print(input_npz_list)

data_list = []
for npz in input_npz_list:
    inputs = np.load(npz)
    key = list(inputs.keys())[0]
    inputs = inputs[key]
    n_files, height, width = inputs.shape
    inputs = inputs.reshape((n_files, -1))
    inputs = scale(inputs, axis=1)
    inputs = inputs.reshape((n_files, height, width))
    print('inputs', inputs.shape)
    data_list.append(inputs)

data_list = np.array(data_list)
data_list = np.concatenate(data_list, axis=0)
print('data_list_shape', data_list.shape)
data_list = data_list.reshape((-1, 256, 256))
len_data_list = np.shape(data_list)[0]
print('len_data_list', len_data_list)
len_chunk = len_data_list // 5
print('len_chunk', len_chunk)

np.savez_compressed(os.path.join(SAVE_PATH, 'input_3dce_feature_extractor_{}.npz'.format(0)), chunk0=data_list[len_chunk*0:len_chunk*1])
np.savez_compressed(os.path.join(SAVE_PATH, 'input_3dce_feature_extractor_{}.npz'.format(1)), chunk1=data_list[len_chunk*1:len_chunk*2])
np.savez_compressed(os.path.join(SAVE_PATH, 'input_3dce_feature_extractor_{}.npz'.format(2)), chunk2=data_list[len_chunk*2:len_chunk*3])
np.savez_compressed(os.path.join(SAVE_PATH, 'input_3dce_feature_extractor_{}.npz'.format(3)), chunk3=data_list[len_chunk*3:len_chunk*4])
np.savez_compressed(os.path.join(SAVE_PATH, 'input_3dce_feature_extractor_{}.npz'.format(4)), chunk4=data_list[len_chunk*4: ])
################################################################
# input = np.load(os.path.join(SAVE_PATH, 'input_3dce.npz'))
# label = np.load(os.path.join(SAVE_PATH, 'label_3dce.npz'))

# input_keys = list(input.keys())
# input_keys = sorted(input_keys)
# print('input_keys', input_keys)
# for key in input_keys:
#     print('shape',input[key][0].shape)
#     print('label',label[key][0])