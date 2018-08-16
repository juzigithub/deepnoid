import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os
import time

import loader
import config as cfg
import performance_eval as pe
from model_f import Model
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

class Task2_label_maker:
    def __init__(self, restore=True):

        self.model = Model()
        self.p_eval = pe.performance()
        self.restore = restore

        if cfg.REBUILD_TASK2_VAL_DATA:
            print('')
            print('>>> Task2 Data Saving Started')
            print('')

            dstime = time.time()
            tl.files.exists_or_mkdir(cfg.SAVE_SURVIVAL_DATA_PATH)

            self.survival_id_list = loader.survival_data_saver(cfg.VAL_DATA_PATH, cfg.VAL_SURVIVAL_CSV_PATH, cfg.SAVE_SURVIVAL_DATA_PATH, train=False)

            detime = time.time()

            print('')
            print('>>> Task2 Data Saving Complete. Consumption Time :', detime - dstime)
            print('')
        else:
            self.survival_id_list = loader.survival_id_extractor(cfg.VAL_SURVIVAL_CSV_PATH, train=False)

        # make paths
        *self.train_start_time, _, _, _, _ = time.localtime()
        self.ckpt_path = '.{0}best{0}'.format(cfg.PATH_SLASH)
        self.log_path = '.{0}logs{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        self._make_path()

        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_path)


    def test(self):
        with tf.Session() as sess:

            #  Saving a model is saving variables such as weights, ans we call it as ckpt(check point file) in tensorflow
            saver = tf.train.Saver(var_list = tf.global_variables())

            # save graphs from tensorboard
            self.writer.add_graph(sess.graph)

            # initialize global variables from session. Need to assign initial values for each variables
            sess.run(tf.global_variables_initializer())

            if self.restore:
                saver.restore(sess, self.ckpt_path + 'brats.ckpt')

            print("BEGIN TESTING")
            task2_X = np.load(cfg.SAVE_SURVIVAL_DATA_PATH + 'task2_val_image.npy')

            task2_et_list = []
            task2_tc_list = []
            task2_wt_list = []
            img_cnt = 0
            survival_id_idx = 0
            print_img_idx = 0

            for batch in tl.iterate.minibatches(inputs=task2_X, targets=task2_X,
                                                batch_size=cfg.N_PATCH_TO_IMG, shuffle=False):
                print_img_idx += 1
                batch_x, _ = batch

                val_feed_dict = {self.model.X: batch_x,
                                 self.model.Y: batch_x,
                                 self.model.training: False,
                                 self.model.drop_rate: 0}

                pred = sess.run(self.model.pred, feed_dict=val_feed_dict)
                pred = utils.reconstruct_from_patches_nd(pred, (cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], cfg.N_CLASS), cfg.PATCH_STRIDE)

                pred = np.argmax(pred, axis=-1)

                pred_list, _ = utils.convert_to_subregions(pred, pred,
                                                           [cfg.ET_LABEL, cfg.TC_LABEL, cfg.WT_LABEL],
                                                           one_hot=False)
                img_cnt += 1
                print('pred_list', np.shape(pred_list))
                task2_et_list.append(pred_list[0])
                task2_tc_list.append(pred_list[1])
                task2_wt_list.append(pred_list[2])
                
                if img_cnt == 155:
                    img_cnt = 0
                    task2_et_list = np.array(task2_et_list).transpose([1,0,2,3,4]).reshape([1,-1,192,192])
                    task2_tc_list = np.array(task2_tc_list).transpose([1,0,2,3,4]).reshape([1,-1,192,192])
                    task2_wt_list = np.array(task2_wt_list).transpose([1,0,2,3,4]).reshape([1,-1,192,192])

                    ########### 192 160 150 ##########
                    # survival_img = np.concatenate([task2_et_list, task2_tc_list, task2_wt_list], axis=0).transpose([0,2,3,1])
                    ##################################

                    ##########  240 240 155 ##########
                    survival_img = np.concatenate([task2_et_list, task2_tc_list, task2_wt_list], axis=0).transpose([0,3,2,1])
                    survival_img = np.pad(survival_img, ((0,0), (24, 24), (24, 24), (0, 0)), 'constant')
                    # survival_img = np.flip(survival_img, axis=1)
                    # survival_img = np.flip(survival_img, axis=2)
                    ##################################

                    task2_et_list = []
                    task2_tc_list = []
                    task2_wt_list = []
                    np.save((cfg.SAVE_SURVIVAL_DATA_PATH + 'validation/{}.npy').format(self.survival_id_list[survival_id_idx]), survival_img)
                    survival_id_idx += 1

            print("TESTING COMPLETED")

    def _make_path(self):
        # create if there is no such file in a saving path
        tl.files.exists_or_mkdir(cfg.SAVE_SURVIVAL_DATA_PATH + 'validation/')

if __name__ == "__main__":
    tester = Task2_label_maker(restore=True)
    tester.test()
