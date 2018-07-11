import numpy as np
import tensorflow as tf
import tensorlayer as tl

import os
# import loadutils
import time
# import resnet
# import deeplab

# from resnet import Model

import loader
import config as cfg
import performance_eval as pe
from model import Model
import utils
import cv2
# import brats2018.utils as utils
# import brats2018.performance_eval as pe
# import brats2018.config as cfg
# import brats2018.loader as loader
# from brats2018.model import Model


os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

class Train:
    def __init__(self):

        # self.data_loader = loader.DataLoader()
        # self.model = resnet.Model()
        self.model = Model()
        self.p_eval = pe.performance()

        if cfg.REBUILD_DATA:
            print('')
            print('>>> Data Saving Started')
            print('')

            dstime = time.time()
            tl.files.exists_or_mkdir(cfg.SAVE_DATA_PATH)

            loader.data_saver([cfg.HGG_DATA_PATH], cfg.SAVE_DATA_PATH, cfg.SPLITS, train=cfg.TRAIN_YN)

            detime = time.time()

            print('')
            print('>>> Data Saving Complete. Consumption Time :', detime - dstime)
            print('')

        # make paths
        *self.train_start_time, _, _, _, _ = time.localtime()
        self.result_txt = '{}_{}_{}_{}_{}.txt'.format(*self.train_start_time)
        self.model_path = '.{0}model{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        self.img_path = '.{0}imgs{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        self.log_path = '.{0}logs{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)

        with open('.{}config.py'.format(cfg.PATH_SLASH), 'rt') as f:
            tl.files.exists_or_mkdir(self.model_path)
            self.result = f.read()
            utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

        # TB
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_path)



        # self.all_X = np.concatenate([np.load(cfg.SAVE_DATA_PATH + 'brats_image_chunk_{}.npy'.format(i)) for i in range(cfg.N_FILES)], axis=0)
        # self.all_Y = np.concatenate([np.load(cfg.SAVE_DATA_PATH + 'brats_label_chunk_{}.npy'.format(i)) for i in range(cfg.N_FILES)], axis=0)
        # self.data_length = np.shape(self.all_X)[0]
        # self.val_data_length = self.data_length // 5


    def optimizer(self, global_step):
        exponential_decay_learning_rate = tf.train.exponential_decay(learning_rate=cfg.INIT_LEARNING_RATE,
                                                                     global_step=global_step,
                                                                     decay_steps=cfg.DECAY_STEP,
                                                                     decay_rate=cfg.DECAY_RATE,
                                                                     staircase=cfg.DECAY_STAIRCASE,
                                                                     name='learning_rate')

        self.optimizer = utils.select_optimizer(cfg.OPTIMIZER, exponential_decay_learning_rate, self.model.loss, global_step)


    def train(self):
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # 배치정규화를 진행하는 경우 배치별 이동평균과 표준편차를 갱신해주는 update operation을 실행하고 지정해줍니다.
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer(global_step)

        with tf.Session() as sess:

            #  Saving a model is saving variables such as weights, ans we call it as ckpt(check point file) in tensorflow
            # It's a tensorflow class saving ckpt file
            saver = tf.train.Saver()

            # save graphs from tensorboard
            self.writer.add_graph(sess.graph)

            # initialize global variables from session. Need to assign initial values for each variables
            sess.run(tf.global_variables_initializer())

            print("BEGIN TRAINING")
            total_training_time = 0

            et_total_result_list = []
            tc_total_result_list = []
            wt_total_result_list = []

            for idx in range(cfg.SPLITS):
                split_start = time.time()
                split_training_time = 0

                train_idx = [i for i in range(cfg.SPLITS) if i != idx]
                # print('train_idx : ', train_idx)
                val_idx = idx
                train_X = np.concatenate(
                    [np.load(cfg.SAVE_DATA_PATH + 'brats_image_chunk_{}.npy'.format(i)) for i in train_idx], axis=0)
                train_Y = np.concatenate(
                    [np.load(cfg.SAVE_DATA_PATH + 'brats_label_chunk_{}.npy'.format(i)) for i in train_idx], axis=0)
                val_X = np.load(cfg.SAVE_DATA_PATH + 'brats_image_chunk_{}.npy'.format(val_idx))
                val_Y = np.load(cfg.SAVE_DATA_PATH + 'brats_label_chunk_{}.npy'.format(val_idx))


                train_step = train_X.shape[0] // cfg.BATCH_SIZE
                # val_step = val_X.shape[0] // cfg.BATCH_SIZE

                et_one_split_result = []
                tc_one_split_result = []
                wt_one_split_result = []

                # print('et_one_split_result : ', et_one_split_result)
                # print('tc_one_split_result : ', tc_one_split_result)
                # print('wt_one_split_result : ', wt_one_split_result)


                for epoch in range(cfg.EPOCHS):
                    epoch_start = time.time()
                    # create variables to save results
                    mean_iou_list, unfiltered_iou_list, loss_list = [], [], []
                    total_cost, total_val_iou, total_val_unfiltered_iou, step = 0, 0, 0, 0

                    # save image and model at the first epoch, final epoch and multiples of SAVING_EPOCH
                    save_yn = (epoch == 0 or epoch + 1 == cfg.EPOCHS or epoch % cfg.SAVING_EPOCH == 0)

                    # Make folder in the saving path for qualified epochs
                    tl.files.exists_or_mkdir('./img/epoch{}/result/'.format(str(epoch + 1)))
                    tl.files.exists_or_mkdir('./img/epoch{}/mask/'.format(str(epoch + 1)))
                    tl.files.exists_or_mkdir('./img/epoch{}/original/'.format(str(epoch + 1)))


                    if save_yn:
                        self._make_path(epoch)
#########################################################
                    # for batch in tl.iterate.minibatches(inputs=train_X, targets=train_Y,
                    #                                     batch_size=cfg.BATCH_SIZE, shuffle=True):
                    #     batch_x, batch_y = batch
                    #
                    #     # make_one_hot
                    #     key = np.array(cfg.TRAIN_LABEL)
                    #     # [0,1,2,3]
                    #     _, index = np.unique(batch_y, return_inverse=True)
                    #     # 4 -> 3
                    #     seg = key[index].reshape(batch_y.shape)
                    #
                    #     batch_y = np.eye(4)[seg]
                    #
                    #
                    #     # step_time = time.time()
                    #     tr_feed_dict = {self.model.X: batch_x,
                    #                     self.model.Y: batch_y,
                    #                     self.model.training: True,
                    #                     self.model.drop_rate: 0.2}
                    #
                    #     # cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)
                    #     cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)
                    #
                    #     bg, ncr, ed, et = sess.run([self.model.bg_loss, self.model.ncr_loss, self.model.ed_loss, self.model.et_loss],
                    #                                feed_dict=tr_feed_dict)
                    #     s = bg + ncr + ed + et
                    #     print('bg', ( bg/s ) * 100)
                    #     print('ncr', ( ncr/s ) * 100)
                    #     print('ed', ( ed/s ) * 100)
                    #     print('et', ( et/s ) * 100)
                    #
                    #     total_cost += cost
                    #     step += 1
                    #
                    #     # print out current epoch, step and batch loss value
                    #     self.result = 'Cross validation : {0} / {1}, Epoch: {2} / {3}, Step: {4} / {5}, Batch loss: {6}'.format((idx + 1),
                    #                                                                                                             cfg.SPLITS,
                    #                                                                                                             epoch + 1,
                    #                                                                                                             cfg.EPOCHS,
                    #                                                                                                             step,
                    #                                                                                                             train_step,
                    #                                                                                                             cost)
                    #
                    #     print(self.result)
                        # utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)
###################################################
                    et_one_epoch_result_list = []
                    tc_one_epoch_result_list = []
                    wt_one_epoch_result_list = []
                    print_img_idx = 0
                    # print_img = 1
                    for batch in tl.iterate.minibatches(inputs=val_X, targets=val_Y,
                                                        batch_size=cfg.BATCH_SIZE, shuffle=False):
                        print_img_idx += 1
                        batch_x, batch_y = batch

                        # make_one_hot
                        key = np.array(cfg.TRAIN_LABEL)
                        _, index = np.unique(batch_y, return_inverse=True)
                        seg = key[index].reshape(batch_y.shape)
                        batch_y = np.eye(len(cfg.TRAIN_LABEL))[seg]



                        val_feed_dict = {self.model.X: batch_x,
                                         self.model.Y: batch_y,
                                         self.model.training: False,
                                         self.model.drop_rate: 0}


                        pred, label = sess.run([self.model.pred, self.model.Y], feed_dict=val_feed_dict)
                        pred = np.argmax(pred, axis=-1)
                        label = np.argmax(label, axis=-1)


                        # label -> pred
                        _, index = np.unique(pred, return_inverse=True)
                        seg = key[index].reshape(pred.shape)
                        pred_print = np.eye(len(cfg.TRAIN_LABEL))[seg]
                        pred_print = np.transpose(pred_print, [-1, 0, 1, 2])


                        pred_list, label_list = utils.convert_to_subregions(pred, label,
                                                                            [cfg.ET_LABEL, cfg.TC_LABEL, cfg.WT_LABEL],
                                                                            one_hot=False)

                        et_one_batch_result = utils.cal_result(pred_list[0], label_list[0], one_hot=False)
                        tc_one_batch_result = utils.cal_result(pred_list[1], label_list[1], one_hot=False)
                        wt_one_batch_result = utils.cal_result(pred_list[2], label_list[2], one_hot=False)

                        et_one_epoch_result_list.append(et_one_batch_result)
                        tc_one_epoch_result_list.append(tc_one_batch_result)
                        wt_one_epoch_result_list.append(wt_one_batch_result)

                        ########################################
                        # image save #
                        # original_img : batch_x / et_img : pred_list[0] / tc_img : pred_list[1] / wt_img : pred_list[2]
                        #################################################
                        # print_img_idx += cfg.BATCH_SIZE
                        # if (print_img_idx >= 60) and print_img==1:
                        #     print_img *= -1
                        #     revert_img_idx = -1 - print_img_idx % 60
                        #
                        #     et_mask = utils.masking_rgb(pred_print[1][revert_img_idx], color='blue')
                        #     tc_mask = utils.masking_rgb(pred_print[2][revert_img_idx], color='red')
                        #     wt_mask = utils.masking_rgb(pred_print[3][revert_img_idx], color='green')
                        #
                        #     p = 0.0001
                        #
                        #     et_tc_mask = cv2.addWeighted(et_mask, float(50) * p, tc_mask, float(50) * p, 0)
                        #     et_tc_wt_mask = cv2.addWeighted(et_tc_mask, float(70) * p, wt_mask, float(30) * p, 0)
                        #     ori = np.transpose(batch_x, [-1, 0, 1, 2])
                        #     ori = utils.masking_rgb(ori[0][revert_img_idx], color=None)
                        #     print('mask', np.shape(et_tc_wt_mask))
                        #     print('ori', np.shape(ori))
                        #     result_image = cv2.addWeighted(ori, float(100 - 40) * p, et_tc_wt_mask, float(60) * p, 0)
                        #     result_image *= 255
                        #
                        #     cv2.imwrite('./et.jpg', et_mask)
                        #     cv2.imwrite('./tc.jpg', tc_mask)
                        #     cv2.imwrite('./wt.jpg', wt_mask)
                        #     cv2.imwrite('./or.jpg', ori)
                        #     cv2.imwrite('./result.jpg', result_image)
                        # if (print_img_idx >= 149) :
                        #     print_img *= -1
                        #     print_img_idx = print_img_idx - 149
                        #################################################
                        p = 0.0001
                        for i in range(0, cfg.BATCH_SIZE, cfg.BATCH_SIZE//2):
                            # et -> ncr
                            ncr_mask = utils.masking_rgb(pred_print[1][i], color='blue')
                            # tc -> ed
                            ed_mask = utils.masking_rgb(pred_print[2][i], color='red')
                            # wt -> et
                            et_mask = utils.masking_rgb(pred_print[3][i], color='green')

                            # et_tc_mask = cv2.addWeighted(et_mask, float(50) * p, tc_mask, float(50) * p, 0) * 255
                            # et_tc_wt_mask = cv2.addWeighted(et_tc_mask, float(50) * p, wt_mask, float(50) * p, 0) * 255
                            # tc -> wt -> et

                            et_tc_wt = ed_mask + 2 * ncr_mask + 3 * et_mask
                            shape = np.shape(et_tc_wt)
                            # et_tc_wt_mask = et_mask + tc_mask + wt_mask
                            et_tc_wt_mask = et_tc_wt.reshape([-1,3])
                            len_mask = len(et_tc_wt_mask)
                            et_tc_wt_mask = et_tc_wt_mask - (0.9*et_tc_wt_mask.max(1).reshape([len_mask, -1]) - et_tc_wt_mask.min(1).reshape([len_mask, -1]))
                            et_tc_wt_mask = np.clip(et_tc_wt_mask, 0., 1.) * 255
                            et_tc_wt_mask = et_tc_wt_mask.reshape(shape)

                            # a = np.array([[1, 2, 3],
                            #               [0, 2, 3],
                            #               [1, 0, 3],
                            #               [0, 0, 3],
                            #               [1, 2, 0]])
                            # s = len(a)
                            # print(s)
                            # b = a - (a.max(1).reshape([s, -1]) - 1)
                            # b = np.clip(b, 0., 1.)
                            # print(b * 255)

                            ori = np.transpose(batch_x, [-1, 0, 1, 2])
                            ori = utils.masking_rgb(ori[0][i], color=None)



                            result_image = cv2.addWeighted(ori, float(100 - 60) * p, et_tc_wt_mask, float(60) * p, 0) * 255

                            cv2.imwrite('./img/epoch{}/result/batch{}_{}.jpg'.format(epoch+1, print_img_idx, i+1), result_image)
                            cv2.imwrite('./img/epoch{}/mask/batch{}_{}_ncr.jpg'.format(epoch+1, print_img_idx, i+1), ncr_mask)
                            cv2.imwrite('./img/epoch{}/mask/batch{}_{}_ed.jpg'.format(epoch+1, print_img_idx, i+1), ed_mask)
                            cv2.imwrite('./img/epoch{}/mask/batch{}_{}_et.jpg'.format(epoch+1, print_img_idx, i+1), et_mask)
                            cv2.imwrite('./img/epoch{}/mask/batch{}_{}_all.jpg'.format(epoch+1, print_img_idx, i+1), et_tc_wt_mask)
                            if print_img_idx == 1:
                                cv2.imwrite('./img/epoch{}/original/batch{}_{}.jpg'.format(epoch+1, print_img_idx, i+1), ori)

                        ########################################


                    et_one_epoch_mean = np.mean(np.array(et_one_epoch_result_list), axis=0)
                    tc_one_epoch_mean = np.mean(np.array(tc_one_epoch_result_list), axis=0)
                    wt_one_epoch_mean = np.mean(np.array(wt_one_epoch_result_list), axis=0)

                    et_one_split_result.append(et_one_epoch_mean)
                    tc_one_split_result.append(tc_one_epoch_mean)
                    wt_one_split_result.append(wt_one_epoch_mean)


                    epoch_end = time.time()
                    training_time = epoch_end - epoch_start
                    split_training_time += training_time
                    total_training_time += training_time

                    Loss = total_cost / train_step
                    print('et_one_epoch_mean', et_one_epoch_mean)
                    print('tc_one_epoch_mean', tc_one_epoch_mean)
                    print('wt_one_epoch_mean', wt_one_epoch_mean)

                    # [acc, sens, spec, miou, dice, hdorff]
                    self.result = ' Cross validation : {} / {}, Epoch: {} / {}, Loss : {}, Training time: {:.2f}' \
                                  '\n1) ET >>> ' \
                                  '\n\t Accuracy : {:.4f}, Sensitivity : {:.4f}, Specificity : {:.4f}' \
                                  '\n\t mean_IoU : {:.4f}, Dice_score : {:.4f}, Hausdorff : {:.4f}' \
                                  '\n2) TC >>> ' \
                                  '\n\t Accuracy : {:.4f}, Sensitivity : {:.4f}, Specificity : {:.4f}' \
                                  '\n\t mean_IoU : {:.4f}, Dice_score : {:.4f}, Hausdorff : {:.4f}' \
                                  '\n3) WT >>> ' \
                                  '\n\t Accuracy : {:.4f}, Sensitivity : {:.4f}, Specificity : {:.4f}' \
                                  '\n\t mean_IoU : {:.4f}, Dice_score : {:.4f}, Hausdorff : {:.4f}'.format((idx+1),
                                                                                                  cfg.SPLITS,
                                                                                                  (epoch + 1),
                                                                                                  cfg.EPOCHS,
                                                                                                  Loss,
                                                                                                  training_time,
                                                                                                  *et_one_epoch_mean,
                                                                                                  *tc_one_epoch_mean,
                                                                                                  *wt_one_epoch_mean)
                    print(self.result)
                    utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

                    result_dict = {self.p_eval.acc: et_one_epoch_mean[0],
                                   self.p_eval.sens: et_one_epoch_mean[1],
                                   self.p_eval.spec: et_one_epoch_mean[2],
                                   self.p_eval.miou: et_one_epoch_mean[3],
                                   self.p_eval.dice: et_one_epoch_mean[4],
                                   self.p_eval.hdorff: et_one_epoch_mean[5]}


                    # TB
                    summary = sess.run(self.merged_summary, feed_dict=result_dict)
                    self.writer.add_summary(summary, global_step=epoch)

                    # 모델 저장
                    if save_yn:
                        saver.save(sess, self.model_save_path)
                        print(">>> Model SAVED")
                        print('')

                et_one_split_mean = np.mean(np.array(et_one_split_result), axis=0)
                tc_one_split_mean = np.mean(np.array(tc_one_split_result), axis=0)
                wt_one_split_mean = np.mean(np.array(wt_one_split_result), axis=0)


                et_total_result_list.append(et_one_split_mean)
                tc_total_result_list.append(tc_one_split_mean)
                wt_total_result_list.append(wt_one_split_mean)




                            # save validation image results
                            # if save_yn:
                            #     self._make_img(predicted_result, x_list, y_list, address, cfg.W, cfg.P)
                self.result = '################# SPLIT #################\n' \
                              'Cross validation : {} / {}, Epoch: {} / {}, Training time: {:.2f}'  \
                              '\n1) ET >>> ' \
                              '\n\t Accuracy : {:.4f}, Sensitivity : {:.4f}, Specificity : {:.4f}' \
                              '\n\t mean_IoU : {:.4f}, Dice_score : {:.4f}, Hausdorff : {:.4f}' \
                              '\n2) TC >>> ' \
                              '\n\t Accuracy : {:.4f}, Sensitivity : {:.4f}, Specificity : {:.4f}' \
                              '\n\t mean_IoU : {:.4f}, Dice_score : {:.4f}, Hausdorff : {:.4f}' \
                              '\n3) WT >>> ' \
                              '\n\t Accuracy : {:.4f}, Sensitivity : {:.4f}, Specificity : {:.4f}' \
                              '\n\t mean_IoU : {:.4f}, Dice_score : {:.4f}, Hausdorff : {:.4f}' \
                              '#########################################'.format((idx + 1),
                                                                                 cfg.SPLITS,
                                                                                 (epoch + 1),
                                                                                 cfg.EPOCHS,
                                                                                 split_training_time,
                                                                                 *et_one_split_mean,
                                                                                 *tc_one_split_mean,
                                                                                 *wt_one_split_mean)
                print(self.result)
                utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

            et_total_mean = np.mean(np.array(et_total_result_list), axis=0)
            et_total_std = np.std(np.array(et_total_result_list), axis=0)
            tc_total_mean = np.mean(np.array(tc_total_result_list), axis=0)
            tc_total_std = np.std(np.array(tc_total_result_list), axis=0)
            wt_total_mean = np.mean(np.array(wt_total_result_list), axis=0)
            wt_total_std = np.std(np.array(wt_total_result_list), axis=0)

            # [acc, sens, spec, miou, dice, hdorff]
            self.result = 'ET >>> ' \
                          '\n\tAccuracy: {:.4f} ± {:.2f} ' \
                          '\n\tSensitivity {:.4f} ± {:.2f} ' \
                          '\n\tSpecificity: {:.4f} ± {:.2f} ' \
                          '\n\tDice Score : {:.4f} ± {:.2f} ' \
                          '\n\tMean IoU : {:.4f} ± {:.2f} ' \
                          '\n\tHausdorff_D : {:.4f} ± {:.2f}'.format(et_total_mean[0], et_total_std[0],
                                                                     et_total_mean[1], et_total_std[1],
                                                                     et_total_mean[2], et_total_std[2],
                                                                     et_total_mean[3], et_total_std[3],
                                                                     et_total_mean[4], et_total_std[4],
                                                                     et_total_mean[5], et_total_std[5])
            utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

            self.result = 'TC >>> ' \
                          '\n\tAccuracy: {:.4f} ± {:.2f} ' \
                          '\n\tSensitivity {:.4f} ± {:.2f} ' \
                          '\n\tSpecificity: {:.4f} ± {:.2f} ' \
                          '\n\tDice Score : {:.4f} ± {:.2f} ' \
                          '\n\tMean IoU : {:.4f} ± {:.2f} ' \
                          '\n\tHausdorff_D : {:.4f} ± {:.2f}'.format(tc_total_mean[0], tc_total_std[0],
                                                                     tc_total_mean[1], tc_total_std[1],
                                                                     tc_total_mean[2], tc_total_std[2],
                                                                     tc_total_mean[3], tc_total_std[3],
                                                                     tc_total_mean[4], tc_total_std[4],
                                                                     tc_total_mean[5], tc_total_std[5])
            utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

            self.result = 'WT >>> ' \
                          '\n\tAccuracy: {:.4f} ± {:.2f} ' \
                          '\n\tSensitivity {:.4f} ± {:.2f} ' \
                          '\n\tSpecificity: {:.4f} ± {:.2f} ' \
                          '\n\tDice Score : {:.4f} ± {:.2f} ' \
                          '\n\tMean IoU : {:.4f} ± {:.2f} ' \
                          '\n\tHausdorff_D : {:.4f} ± {:.2f}'.format(wt_total_mean[0], wt_total_std[0],
                                                                     wt_total_mean[1], wt_total_std[1],
                                                                     wt_total_mean[2], wt_total_std[2],
                                                                     wt_total_mean[3], wt_total_std[3],
                                                                     wt_total_mean[4], wt_total_std[4],
                                                                     wt_total_mean[5], wt_total_std[5])
            utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

            # print('TC >>> Accuracy: {}, Sensitivity {}, Specificity: {}, Dice Score : {}, Mean IoU : {}, Hausdorff_D : {}'.format(tc_total_mean[0],
            #                                                                                                                       tc_total_mean[1],
            #                                                                                                                       tc_total_mean[2],
            #                                                                                                                       tc_total_mean[3],
            #                                                                                                                       tc_total_mean[4],
            #                                                                                                                       tc_total_mean[5]))
            #
            # print('WT >>> Accuracy: {}, Sensitivity {}, Specificity: {}, Dice Score : {}, Mean IoU : {}, Hausdorff_D : {}'.format(wt_total_mean[0],
            #                                                                                                                       wt_total_mean[1],
            #                                                                                                                       wt_total_mean[2],
            #                                                                                                                       wt_total_mean[3],
            #                                                                                                                       wt_total_mean[4],
            #                                                                                                                       wt_total_mean[5]))


#                     train_idx = [i for i in range(cfg.SPLITS) if i != idx1]
#                     val_idx = [idx1]
#                     for idx2 in range(cfg.SUB_SPLITS):
#
#
#                         train_X = np.concatenate([np.load(cfg.SAVE_DATA_PATH + 'brats_image_chunk_{}.npy'.format(cfg.SUB_SPLITS * i + idx2)) for i in train_idx], axis=0)
#                         train_Y = np.concatenate([np.load(cfg.SAVE_DATA_PATH + 'brats_label_chunk_{}.npy'.format(cfg.SUB_SPLITS * i + idx2)) for i in train_idx], axis=0)
#                         val_X = np.concatenate([np.load(cfg.SAVE_DATA_PATH + 'brats_image_chunk_{}.npy'.format(cfg.SUB_SPLITS * i + idx2)) for i in val_idx], axis=0)
#                         val_Y = np.concatenate([np.load(cfg.SAVE_DATA_PATH + 'brats_label_chunk_{}.npy'.format(cfg.SUB_SPLITS * i + idx2)) for i in val_idx], axis=0)
#
#                         train_step = train_X.shape[0] // cfg.BATCH_SIZE
#                         val_step = val_X.shape[0] // cfg.BATCH_SIZE
#
#                         # shuffle
#
#                         # create variables to save results
#                         mean_iou_list, unfiltered_iou_list, loss_list = [], [], []
#                         total_cost, total_val_iou, total_val_unfiltered_iou, step = 0, 0, 0, 0
#
#                         # for bath in range(train_step):
#                         for batch in tl.iterate.minibatches(inputs=train_X, targets=train_Y,
#                                                             batch_size=cfg.BATCH_SIZE, shuffle=True):
#
#                             batch_x, batch_y = batch
#                             # step_time = time.time()
#
# ######################################################
#                             tr_feed_dict = {self.model.X: batch_x,
#                                             self.model.Y: batch_y,
#                                             self.model.training: True,
#                                             self.model.drop_rate: 0.2}
#
#                             cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)
#
#                             total_cost += cost
#                             step += 1
#
#                             # print out current epoch, step and batch loss value
#                             # self.result = 'Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % cfg.EPOCHS, 'Step:', step, '/', train_step,'  Batch loss:', cost
#                             self.result = 'Epoch: {0} / {1}, Sub splits : {2} / {3}, Step: {4} / {5}, Batch loss: {6}'.format((epoch + 1),
#                                                                                                                               cfg.EPOCHS,
#                                                                                                                               idx2,
#                                                                                                                               cfg.SUB_SPLITS,
#                                                                                                                               step,
#                                                                                                                               train_step,
#                                                                                                                               cost)
#
#                             print(self.result)
#                             utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)
# ######################################################
#
#                         for batch in tl.iterate.minibatches(inputs=val_X, targets=val_Y,
#                                                             batch_size=cfg.BATCH_SIZE, shuffle=True):
#
#                             batch_x, batch_y = batch
#                             # step_time = time.time()
#
# ###################################################
#                             # feed_dict for iterator
#                             val_feed_dict = {self.model.X: batch_x,
#                                              self.model.Y: batch_y,
#                                              self.model.training: False,
#                                              self.model.drop_rate: 0}
#
#
#                             # Calculate validation Iou(Intersection of Union). Iou is used as an accuracy in image segmentation.
#                             # return [acc, mean_iou, unfiltered_iou] in model.iou
#                             val_results, predicted_result, x_list, y_list = sess.run([self.model.results,
#                                                                                       self.model.logit,
#                                                                                       self.model.X,
#                                                                                       self.model.Y],
#                                                                                      feed_dict=val_feed_dict)
#                             # acc, val_mean_iou, val_unfiltered_iou = val_results
#
#                             # convert received batch iou as a list
#                             ious = list(val_results[0])
#                             unfiltered_iou = np.mean(ious)
#
#                             # uses only iou > 0.01 (i.e. IoUs predicting over certain cutline) to calculate IoUs for diagnosis accuracy
#                             iou_list = []
#
#                             for iou in ious:
#                                 if iou > 0.01:
#                                     iou_list.append(iou)
#
#                             after_filtered_length = len(iou_list)
#                             # before_filtered_length = len(ious)
#
#                             # val_batch_acc = after_filtered_length / before_filtered_length
#
#                             if after_filtered_length == 0:
#                                 mean_iou = 0
#
#                             else:
#                                 mean_iou = np.mean(iou_list)
#
#                             # Add IoUs per patch and accuracies to entire IoU value. As epoch terminated, convert to average IoU and ave accuracy.
#                             total_val_iou += mean_iou
#                             total_val_unfiltered_iou += unfiltered_iou
#
#                             # save validation image results
#                             # if save_yn:
#                             #     self._make_img(predicted_result, x_list, y_list, address, cfg.W, cfg.P)

###################################################

    def _make_path(self, epoch):
        ### Savin A Model ###

        # Absolute path for model saving. save as 'file_name'.ckpt
        self.model_save_path = self.model_path + '{0}{1}{0}brats.ckpt'.format(cfg.PATH_SLASH,
                                                                             str(epoch + 1))

        # create if there is no such file in a saving path
        tl.files.exists_or_mkdir(self.model_path + '{0}{1}'.format(cfg.PATH_SLASH, str(epoch + 1)))

        ### Save validation image result ###

        # val_img_save_path overlaps training image(original image) and predicted image and overlays mask image at affected area
        # raw_val_img_save_path is saving path for predicted image and label_val_img_save_path is for label image
        # save image and model at the first epoch, final epoch and multiples of SAVING_EPOCH

        # dir_name = ['merged', 'pred', 'label', 'compare']
        # self.path_list = [(self.img_path + '{0}{1}{0}' + name).format(cfg.PATH_SLASH, str(epoch + 1)) for name in dir_name]

if __name__ == "__main__":
    trainer = Train()
    trainer.train()
