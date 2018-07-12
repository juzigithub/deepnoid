import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os
import time

import loader
import config as cfg
import performance_eval as pe
from model import Model
import utils
#
# import brats2018.utils as utils
# import brats2018.performance_eval as pe
# import brats2018.config as cfg
# import brats2018.loader as loader
# from brats2018.model import Model

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU


class Task2_label_maker:
    def __init__(self, restore=True):

        self.model = Model()
        self.p_eval = pe.performance()
        self.restore = restore

        if cfg.REBUILD_TASK1_VAL_DATA:
            print('')
            print('>>> Task1 Data Saving Started')
            print('')

            dstime = time.time()
            tl.files.exists_or_mkdir(cfg.SAVE_VALID_DATA_PATH)

            loader.data_saver([cfg.VAL_DATA_PATH], cfg.SAVE_VALID_DATA_PATH, cfg.SPLITS, train=False, shuffle=False)

            detime = time.time()

            print('')
            print('>>> Task1 Data Saving Complete. Consumption Time :', detime - dstime)
            print('')

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
        self.result_txt = '{}_{}_{}_{}_{}.txt'.format(*self.train_start_time)
        self.model_path = '.{0}model{0}train{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        self.ckpt_path = '.{0}best{0}'.format(cfg.PATH_SLASH)
        # self.img_path = '.{0}imgs{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        self.log_path = '.{0}logs{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        self._make_path()

        # with open('.{}config.py'.format(cfg.PATH_SLASH), 'rt') as f:
        #     self._make_path()
        #     self.result = f.read()
        #     utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

        # TB
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_path)


    def test(self):
        with tf.Session() as sess:

            #  Saving a model is saving variables such as weights, ans we call it as ckpt(check point file) in tensorflow
            # It's a tensorflow class saving ckpt file
            saver = tf.train.Saver()

            # save graphs from tensorboard
            self.writer.add_graph(sess.graph)

            # initialize global variables from session. Need to assign initial values for each variables
            sess.run(tf.global_variables_initializer())

            if self.restore:
                saver.restore(sess, self.ckpt_path + 'brats.ckpt')

            print("BEGIN TESTING")
            total_training_time = 0

            # et_total_result_list = []
            # tc_total_result_list = []
            # wt_total_result_list = []
            ##############################################################
            ################### for task2 ################################
            task2_X = np.load(cfg.SAVE_SURVIVAL_DATA_PATH + 'task2_val_image.npy')

            ################################################# batch_size = 30 -> cfg.BATCH_SIZE 를 150의 약수로!
            task2_et_list = []
            task2_tc_list = []
            task2_wt_list = []
            img_cnt = 0
            survival_id_idx = 0
            print_img_idx = 0
            for batch in tl.iterate.minibatches(inputs=task2_X, targets=task2_X,
                                                batch_size=30, shuffle=False):
                print_img_idx += 1
                batch_x, _ = batch

                val_feed_dict = {self.model.X: batch_x,
                                 self.model.Y: batch_x,
                                 self.model.training: False,
                                 self.model.drop_rate: 0}

                pred = sess.run([self.model.pred], feed_dict=val_feed_dict)
                pred = np.argmax(pred, axis=-1)

                pred_list, _ = utils.convert_to_subregions(pred, pred,
                                                           [cfg.ET_LABEL, cfg.TC_LABEL, cfg.WT_LABEL],
                                                           one_hot=False)
                img_cnt += 30

                task2_et_list.append(pred_list[0])
                task2_tc_list.append(pred_list[1])
                task2_wt_list.append(pred_list[2])

                if img_cnt >= 150:
                    img_cnt = 0
                    survival_img = np.concatenate([[task2_et_list], [task2_tc_list], [task2_wt_list]], axis=0).transpose([0,2,3,1])
                    task2_et_list = []
                    task2_tc_list = []
                    task2_wt_list = []
                    np.save('./img/test/task2/survival/{}.npy'.format(self.survival_id_list[survival_id_idx]), survival_img)


    def _make_path(self):
        # create if there is no such file in a saving path
        tl.files.exists_or_mkdir('./img/test/task2/survival/')

if __name__ == "__main__":
    tester = Task2_label_maker(restore=True)
    tester.test()
