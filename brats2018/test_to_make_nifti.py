import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os
import time
import cv2
import loader
import config as cfg
import performance_eval as pe
from model import Model
import utils


os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU


class Test:
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

        # make paths
        *self.train_start_time, _, _, _, _ = time.localtime()
        self.result_txt = '{}_{}_{}_{}_{}.txt'.format(*self.train_start_time)
        self.model_path = '.{0}model{0}test{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        self.ckpt_path = '.{0}best{0}'.format(cfg.PATH_SLASH)
        self.log_path = '.{0}logs{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        patient_path_list = tl.files.load_folder_list(cfg.VAL_DATA_PATH)
        self.patient_id_list = [os.path.basename(p) for p in patient_path_list]
        self._make_path()

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

            test_X = np.load(cfg.SAVE_VALID_DATA_PATH + 'brats_val_image.npy')

            print_img_idx = 0

            for batch in tl.iterate.minibatches(inputs=test_X, targets=test_X,
                                                batch_size=cfg.BATCH_SIZE, shuffle=False):
                batch_start = time.time()
                batch_x, _ = batch

                key = np.array([0, 1, 2, 4])

                test_feed_dict = {self.model.X: batch_x,
                                 self.model.Y: batch_x,
                                 self.model.training: False,
                                 self.model.drop_rate: 0}

                pred = sess.run(self.model.pred, feed_dict=test_feed_dict)
                pred = np.argmax(pred, axis=-1)
                print(pred)
                pred[pred == 3] = 4


                # _, index = np.unique(pred, return_inverse=True)
                # seg = key[index].reshape(pred.shape)
                # pred_print = np.eye(len(cfg.TRAIN_LABEL))[seg]
                # print('np.shape(prediction) : ',np.shape(pred_print))
                # pred_print = np.transpose(pred_print, [-1, 0, 1, 2])

                # pred_list, _ = utils.convert_to_subregions(pred, pred,
                #                                                     [cfg.ET_LABEL, cfg.TC_LABEL, cfg.WT_LABEL],
                #                                                     one_hot=False)

                print_img_idx += 1

                # for i in range(0, cfg.BATCH_SIZE):
                #     ncr_mask = utils.masking_rgb(pred_print[1][i], color='green')
                #     ed_mask = utils.masking_rgb(pred_print[2][i], color='blue')
                #     et_mask = utils.masking_rgb(pred_print[3][i], color='red')
                #
                #     et_tc_wt = ed_mask + 2 * ncr_mask + 3 * et_mask
                #     shape = np.shape(et_tc_wt)
                #     et_tc_wt_mask = et_tc_wt.reshape([-1, 3])
                #     len_mask = len(et_tc_wt_mask)
                #     et_tc_wt_mask = et_tc_wt_mask - (
                #                 0.9 * et_tc_wt_mask.max(1).reshape([len_mask, -1]) - et_tc_wt_mask.min(1).reshape(
                #             [len_mask, -1]))
                #     et_tc_wt_mask = np.clip(et_tc_wt_mask, 0., 1.) * 255
                #     et_tc_wt_mask = et_tc_wt_mask.reshape(shape)
                #
                #     ori = np.transpose(batch_x, [-1, 0, 1, 2])
                #     ori = utils.masking_rgb(ori[0][i], color=None)
                #
                #     result_image = 0.7 * (ori + et_tc_wt_mask)
                #
                #     cv2.imwrite('./img/test/for_nifti/__________________batch{}_{}.jpg'.format(print_img_idx, i + 1), result_image)

                batch_end = time.time()
                testing_time = batch_end - batch_start
                total_training_time += testing_time

            print("TESTING COMPLETED")

    def _make_path(self):
        # create if there is no such file in a saving path
        tl.files.exists_or_mkdir(self.model_path + '{0}'.format(cfg.PATH_SLASH))
        tl.files.exists_or_mkdir('./img/test/for_nifti/')
        tl.files.exists_or_mkdir(self.model_path)
        tl.files.exists_or_mkdir(self.ckpt_path)

if __name__ == "__main__":
    tester = Test(restore=True)
    tester.test()
