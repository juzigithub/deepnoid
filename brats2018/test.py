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

# import brats2018.utils as utils
# import brats2018.performance_eval as pe
# import brats2018.config as cfg
# import brats2018.loader as loader
# from brats2018.model import Model

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

        if cfg.REBUILD_TASK1_VAL_DATA:
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
            self.survival_id_list = loader.survival_id_extractor(cfg.VAL_SURVIVAL_CSV_PATH)

        # make paths
        *self.train_start_time, _, _, _, _ = time.localtime()
        self.result_txt = '{}_{}_{}_{}_{}.txt'.format(*self.train_start_time)
        self.model_path = '.{0}model{0}test{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        self.ckpt_path = '.{0}best{0}'.format(cfg.PATH_SLASH)
        # self.img_path = '.{0}imgs{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        self.log_path = '.{0}logs{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)

        with open('.{}config.py'.format(cfg.PATH_SLASH), 'rt') as f:
            self._make_path()
            self.result = f.read()
            utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

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

            print("BEGIN TRAINING")
            total_training_time = 0

            et_total_result_list = []
            tc_total_result_list = []
            wt_total_result_list = []
            ##############################################################
            ################### for task2 ################################
            # task2_X = np.load(cfg.SAVE_SURVIVAL_DATA_PATH + 'task2_train_image.npy')
            # task2_Y = np.load(cfg.SAVE_SURVIVAL_DATA_PATH + 'task2_train_label.npy')


            ##############################################################
            # split_training_time = 0
            test_idx = [i for i in range(cfg.SPLITS)]

            test_X = np.concatenate(
                [np.load(cfg.SAVE_VALID_DATA_PATH + 'brats_image_chunk_{}.npy'.format(i)) for i in test_idx], axis=0)


            # test_step = test_X.shape[0] // cfg.BATCH_SIZE
            # print('et_one_split_result : ', et_one_split_result)
            # print('tc_one_split_result : ', tc_one_split_result)
            # print('wt_one_split_result : ', wt_one_split_result)
            print_img_idx = 0
#################################################################
            for batch in tl.iterate.minibatches(inputs=test_X, targets=test_X,
                                                batch_size=cfg.BATCH_SIZE, shuffle=False):
                batch_start = time.time()
                #
                # et_one_batch_result_list = []
                # tc_one_batch_result_list = []
                # wt_one_batch_result_list = []

                # print_img_idx += 1
                batch_x, _ = batch

                key = np.array(cfg.TRAIN_LABEL)

                test_feed_dict = {self.model.X: batch_x,
                                 self.model.Y: batch_x,
                                 self.model.training: False,
                                 self.model.drop_rate: 0}

                pred = sess.run([self.model.pred], feed_dict=test_feed_dict)
                pred = np.argmax(pred, axis=-1)

                # label -> pred
                _, index = np.unique(pred, return_inverse=True)
                seg = key[index].reshape(pred.shape)
                pred_print = np.eye(len(cfg.TRAIN_LABEL))[seg]
                pred_print = np.transpose(pred_print, [-1, 0, 1, 2])

                pred_list, _ = utils.convert_to_subregions(pred, pred,
                                                                    [cfg.ET_LABEL, cfg.TC_LABEL, cfg.WT_LABEL],
                                                                    one_hot=False)

                # et_one_batch_result = utils.cal_result(pred_list[0], label_list[0], one_hot=False)
                # tc_one_batch_result = utils.cal_result(pred_list[1], label_list[1], one_hot=False)
                # wt_one_batch_result = utils.cal_result(pred_list[2], label_list[2], one_hot=False)
                #
                # et_one_batch_result_list.append(et_one_batch_result)
                # tc_one_batch_result_list.append(tc_one_batch_result)
                # wt_one_batch_result_list.append(wt_one_batch_result)

                print_img_idx += 1

                for i in range(0, cfg.BATCH_SIZE):
                    # et -> ncr
                    ncr_mask = utils.masking_rgb(pred_print[1][i], color='blue')
                    # tc -> ed
                    ed_mask = utils.masking_rgb(pred_print[2][i], color='red')
                    # wt -> et
                    et_mask = utils.masking_rgb(pred_print[3][i], color='green')

                    et_tc_wt = ed_mask + 2 * ncr_mask + 3 * et_mask
                    shape = np.shape(et_tc_wt)
                    # et_tc_wt_mask = et_mask + tc_mask + wt_mask
                    et_tc_wt_mask = et_tc_wt.reshape([-1, 3])
                    len_mask = len(et_tc_wt_mask)
                    et_tc_wt_mask = et_tc_wt_mask - (
                                0.9 * et_tc_wt_mask.max(1).reshape([len_mask, -1]) - et_tc_wt_mask.min(1).reshape(
                            [len_mask, -1]))
                    et_tc_wt_mask = np.clip(et_tc_wt_mask, 0., 1.) * 255
                    et_tc_wt_mask = et_tc_wt_mask.reshape(shape)

                    ori = np.transpose(batch_x, [-1, 0, 1, 2])
                    ori = utils.masking_rgb(ori[0][i], color=None)

                    # result_image = cv2.addWeighted(ori, 0.0005, et_tc_wt_mask, 0.1, 0) * 255
                    result_image = 0.5 * (ori + et_tc_wt_mask)

                    cv2.imwrite('./img/test/result/batch{}_{}.jpg'.format(print_img_idx, i + 1), result_image)
                    cv2.imwrite('./img/test/mask/batch{}_{}_ncr.jpg'.format(print_img_idx, i + 1), ncr_mask)
                    cv2.imwrite('./img/test/mask/batch{}_{}_ed.jpg'.format(print_img_idx, i + 1), ed_mask)
                    cv2.imwrite('./img/test/mask/batch{}_{}_et.jpg'.format(print_img_idx, i + 1), et_mask)
                    cv2.imwrite('./img/test/mask/batch{}_{}_all.jpg'.format(print_img_idx, i + 1), et_tc_wt_mask)
                    cv2.imwrite('./img/test/original/batch{}_{}.jpg'.format(print_img_idx, i + 1), ori)
                ########################################

###################################################

                # et_one_batch_mean = np.mean(np.array(et_one_batch_result_list), axis=0)
                # tc_one_batch_mean = np.mean(np.array(tc_one_batch_result_list), axis=0)
                # wt_one_batch_mean = np.mean(np.array(wt_one_batch_result_list), axis=0)
                #
                # et_total_result_list.append(et_one_batch_mean)
                # tc_total_result_list.append(tc_one_batch_mean)
                # wt_total_result_list.append(wt_one_batch_mean)


                batch_end = time.time()
                testing_time = batch_end - batch_start
                total_training_time += testing_time

            #     # [acc, sens, spec, miou, dice, hdorff]
            #     self.result = 'Batch : {} / {}, Training time: {:.2f}' \
            #                   '\n1) ET >>> ' \
            #                   '\n\t Accuracy : {:.4f}, Sensitivity : {:.4f}, Specificity : {:.4f}' \
            #                   '\n\t mean_IoU : {:.4f}, Dice_score : {:.4f}, Hausdorff : {:.4f}' \
            #                   '\n2) TC >>> ' \
            #                   '\n\t Accuracy : {:.4f}, Sensitivity : {:.4f}, Specificity : {:.4f}' \
            #                   '\n\t mean_IoU : {:.4f}, Dice_score : {:.4f}, Hausdorff : {:.4f}' \
            #                   '\n3) WT >>> ' \
            #                   '\n\t Accuracy : {:.4f}, Sensitivity : {:.4f}, Specificity : {:.4f}' \
            #                   '\n\t mean_IoU : {:.4f}, Dice_score : {:.4f}, Hausdorff : {:.4f}'.format(print_img_idx,
            #                                                                                            test_step,
            #                                                                                            testing_time,
            #                                                                                            *et_one_batch_mean,
            #                                                                                            *tc_one_batch_mean,
            #                                                                                            *wt_one_batch_mean)
            #     print(self.result)
            #     utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)
            #
            #     result_dict = {self.p_eval.acc: et_one_batch_mean[0],
            #                    self.p_eval.sens: et_one_batch_mean[1],
            #                    self.p_eval.spec: et_one_batch_mean[2],
            #                    self.p_eval.miou: et_one_batch_mean[3],
            #                    self.p_eval.dice: et_one_batch_mean[4],
            #                    self.p_eval.hdorff: et_one_batch_mean[5]}
            #
            #
            #     # TB
            #     summary = sess.run(self.merged_summary, feed_dict=result_dict)
            #     self.writer.add_summary(summary, global_step=print_img_idx)
            #
            # et_total_mean = np.mean(np.array(et_total_result_list), axis=0)
            # et_total_std = np.std(np.array(et_total_result_list), axis=0)
            # tc_total_mean = np.mean(np.array(tc_total_result_list), axis=0)
            # tc_total_std = np.std(np.array(tc_total_result_list), axis=0)
            # wt_total_mean = np.mean(np.array(wt_total_result_list), axis=0)
            # wt_total_std = np.std(np.array(wt_total_result_list), axis=0)
            #
            # # [acc, sens, spec, miou, dice, hdorff]
            # self.result = 'ET >>> ' \
            #               '\n\tAccuracy: {:.4f} ± {:.2f} ' \
            #               '\n\tSensitivity {:.4f} ± {:.2f} ' \
            #               '\n\tSpecificity: {:.4f} ± {:.2f} ' \
            #               '\n\tDice Score : {:.4f} ± {:.2f} ' \
            #               '\n\tMean IoU : {:.4f} ± {:.2f} ' \
            #               '\n\tHausdorff_D : {:.4f} ± {:.2f}'.format(et_total_mean[0], et_total_std[0],
            #                                                          et_total_mean[1], et_total_std[1],
            #                                                          et_total_mean[2], et_total_std[2],
            #                                                          et_total_mean[3], et_total_std[3],
            #                                                          et_total_mean[4], et_total_std[4],
            #                                                          et_total_mean[5], et_total_std[5])
            # utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)
            #
            # self.result = 'TC >>> ' \
            #               '\n\tAccuracy: {:.4f} ± {:.2f} ' \
            #               '\n\tSensitivity {:.4f} ± {:.2f} ' \
            #               '\n\tSpecificity: {:.4f} ± {:.2f} ' \
            #               '\n\tDice Score : {:.4f} ± {:.2f} ' \
            #               '\n\tMean IoU : {:.4f} ± {:.2f} ' \
            #               '\n\tHausdorff_D : {:.4f} ± {:.2f}'.format(tc_total_mean[0], tc_total_std[0],
            #                                                          tc_total_mean[1], tc_total_std[1],
            #                                                          tc_total_mean[2], tc_total_std[2],
            #                                                          tc_total_mean[3], tc_total_std[3],
            #                                                          tc_total_mean[4], tc_total_std[4],
            #                                                          tc_total_mean[5], tc_total_std[5])
            # utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)
            #
            # self.result = 'WT >>> ' \
            #               '\n\tAccuracy: {:.4f} ± {:.2f} ' \
            #               '\n\tSensitivity {:.4f} ± {:.2f} ' \
            #               '\n\tSpecificity: {:.4f} ± {:.2f} ' \
            #               '\n\tDice Score : {:.4f} ± {:.2f} ' \
            #               '\n\tMean IoU : {:.4f} ± {:.2f} ' \
            #               '\n\tHausdorff_D : {:.4f} ± {:.2f}'.format(wt_total_mean[0], wt_total_std[0],
            #                                                          wt_total_mean[1], wt_total_std[1],
            #                                                          wt_total_mean[2], wt_total_std[2],
            #                                                          wt_total_mean[3], wt_total_std[3],
            #                                                          wt_total_mean[4], wt_total_std[4],
            #                                                          wt_total_mean[5], wt_total_std[5])
            # utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)
            #

            ################################################# batch_size = 30 -> cfg.BATCH_SIZE 를 150의 약수로!
            # task2_et_list = []
            # task2_tc_list = []
            # task2_wt_list = []
            # img_cnt = 0
            # survival_id_idx = 0
            # for batch in tl.iterate.minibatches(inputs=task2_X, targets=task2_Y,
            #                                     batch_size=30, shuffle=False):
            #     print_img_idx += 1
            #     batch_x, batch_y = batch
            #
            #     # make_one_hot
            #     key = np.array(cfg.TRAIN_LABEL)
            #     _, index = np.unique(batch_y, return_inverse=True)
            #     seg = key[index].reshape(batch_y.shape)
            #     batch_y = np.eye(len(cfg.TRAIN_LABEL))[seg]
            #
            #
            #
            #     val_feed_dict = {self.model.X: batch_x,
            #                      self.model.Y: batch_y,
            #                      self.model.training: False,
            #                      self.model.drop_rate: 0}
            #
            #
            #     pred, label = sess.run([self.model.pred, self.model.Y], feed_dict=val_feed_dict)
            #     pred = np.argmax(pred, axis=-1)
            #     label = np.argmax(label, axis=-1)
            #
            #     pred_list, _ = utils.convert_to_subregions(pred, label,
            #                                                [cfg.ET_LABEL, cfg.TC_LABEL, cfg.WT_LABEL],
            #                                                one_hot=False)
            #     img_cnt += 30
            #
            #     task2_et_list.append(pred_list[0])
            #     task2_tc_list.append(pred_list[1])
            #     task2_wt_list.append(pred_list[2])
            #
            #     if img_cnt >= 150:
            #         img_cnt = 0
            #         survival_img = np.concatenate([[task2_et_list], [task2_tc_list], [task2_wt_list]], axis=0).transpose([0,2,3,1])
            #         task2_et_list = []
            #         task2_tc_list = []
            #         task2_wt_list = []
            #         np.save('./img/test/task2/survival/{}.npy'.format(self.survival_id_list[survival_id_idx]), survival_img)
            ##################################################################################


    def _make_path(self):
        # create if there is no such file in a saving path
        tl.files.exists_or_mkdir(self.model_path + '{0}'.format(cfg.PATH_SLASH))
        tl.files.exists_or_mkdir('./img/test/result/')
        tl.files.exists_or_mkdir('./img/test/mask/')
        tl.files.exists_or_mkdir('./img/test/survival/')
        tl.files.exists_or_mkdir('./img/test/original/')
        tl.files.exists_or_mkdir('./img/test/task2/survival/')
        tl.files.exists_or_mkdir(self.model_path)
        tl.files.exists_or_mkdir(self.ckpt_path)

if __name__ == "__main__":
    tester = Test(restore=True)
    tester.test()
