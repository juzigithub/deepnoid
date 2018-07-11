import numpy as np
import config as cfg
import loader
import utils
import tensorlayer as tl

# import brats2018.loader as loader
# import brats2018.config as cfg
# import brats2018.utils as utils
if cfg.REBUILD_TASK2_DATA:
    tl.files.exists_or_mkdir(cfg.SAVE_SURVIVAL_DATA_PATH)
    survival_id_list = loader.survival_data_saver(cfg.HGG_DATA_PATH, cfg.SURVIVAL_CSV_PATH, cfg.SAVE_SURVIVAL_DATA_PATH, train=cfg.TRAIN_YN)
else :
    survival_id_list = loader.survival_id_extractor(cfg.SURVIVAL_CSV_PATH)

task2_Y = np.load(cfg.SAVE_SURVIVAL_DATA_PATH + 'task2_train_label.npy')

key = np.array(cfg.TRAIN_LABEL)
_, index = np.unique(task2_Y, return_inverse=True)
seg = key[index].reshape(task2_Y.shape)
label = np.eye(len(cfg.TRAIN_LABEL))[seg]
label = np.argmax(label, axis=-1)

_, label = utils.convert_to_subregions(label, label,
                                       [cfg.ET_LABEL, cfg.TC_LABEL, cfg.WT_LABEL],
                                       one_hot=False)
label = np.array(label)
_, length, _, _ = np.shape(label)
length //= 150
for idx in range(length):
    survival_img = label[:,150*idx:150*(idx+1),:,:].transpose([0, 2, 3, 1])
    print(np.shape(survival_img))
    np.save(cfg.SAVE_SURVIVAL_DATA_PATH + '{}.npy'.format(survival_id_list[idx]), survival_img)

