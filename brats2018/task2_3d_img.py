import numpy as np
import config as cfg
import loader
import utils
#
# import brats2018.loader as loader
# import brats2018.config as cfg
# import brats2018.utils as utils

if cfg.REBUILD_TASK2_DATA:
    survival_id_list = loader.survival_data_saver([cfg.HGG_DATA_PATH], cfg.SURVIVAL_CSV_PATH, cfg.SAVE_SURVIVAL_DATA_PATH, train=cfg.TRAIN_YN)
task2_Y = np.load(cfg.SAVE_SURVIVAL_DATA_PATH + 'task2_train_label.npy')

key = np.array(cfg.TRAIN_LABEL)
_, index = np.unique(task2_Y, return_inverse=True)
seg = key[index].reshape(task2_Y.shape)
label = np.eye(len(cfg.TRAIN_LABEL))[seg]
label = np.argmax(label, axis=-1)

_, label = utils.convert_to_subregions(label, label,
                                       [cfg.ET_LABEL, cfg.TC_LABEL, cfg.WT_LABEL],
                                       one_hot=False)

print('label', np.shape(label))
print('et', np.shape(label[0]))

#
# for idx in range()
#     img_cnt = 0
#     # np.array(a).transpose([0, 2, 3, 1])
#     survival_img = np.concatenate([[task2_et_list], [task2_tc_list], [task2_wt_list]], axis=0).transpose([0, 2, 3, 1])
#     task2_et_list = []
#     task2_tc_list = []
#     task2_wt_list = []
#     np.save('./img/epoch{}/survival/{}.npy'.format(epoch + 1, self.survival_id_list[survival_id_idx]), survival_img)