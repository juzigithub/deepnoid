import numpy as np
import config as cfg
import loader
import utils
import tensorlayer as tl

if cfg.REBUILD_TASK2_DATA:
    tl.files.exists_or_mkdir(cfg.SAVE_SURVIVAL_DATA_PATH + 'train/')
    tl.files.exists_or_mkdir(cfg.SAVE_SURVIVAL_DATA_PATH)
    survival_id_list = loader.survival_data_saver(cfg.HGG_DATA_PATH, cfg.SURVIVAL_CSV_PATH, cfg.SAVE_SURVIVAL_DATA_PATH, train=cfg.TRAIN_YN)
else :
    tl.files.exists_or_mkdir(cfg.SAVE_SURVIVAL_DATA_PATH + 'train/')
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
    survival_img = label[:,150*idx:150*(idx+1),:,:].transpose([0, 3, 2, 1])
    # survival_img = np.pad(survival_img, ((0, 0), (41, 39), (30, 18), (3, 2)), 'constant')
    # survival_img = np.flip(survival_img, axis=1)
    # survival_img = np.flip(survival_img, axis=2)
    print(np.shape(survival_img))
    np.save(cfg.SAVE_SURVIVAL_DATA_PATH + 'train/{}.npy'.format(survival_id_list[idx]), survival_img)

