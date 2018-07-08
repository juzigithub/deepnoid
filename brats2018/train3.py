import numpy as np
import tensorflow as tf
import tensorlayer as tl

import os
# import loadutils
import time
import utils
# import resnet
# import deeplab


import loader
import config as cfg
from model import Model
import performance_eval as pe
#
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

            loader.data_saver([cfg.HGG_DATA_PATH], cfg.SAVE_DATA_PATH, cfg.N_FILES, train=cfg.TRAIN_YN)

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



        self.all_X = np.concatenate([np.load(cfg.SAVE_DATA_PATH + 'brats_image_chunk_{}.npy'.format(i)) for i in range(cfg.N_FILES)], axis=0)
        self.all_Y = np.concatenate([np.load(cfg.SAVE_DATA_PATH + 'brats_label_chunk_{}.npy'.format(i)) for i in range(cfg.N_FILES)], axis=0)
        self.data_length = np.shape(self.all_X)[0]
        self.val_data_length = self.data_length // 5


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

            for epoch in range(cfg.EPOCHS):
                # save image and model at the first epoch, final epoch and multiples of SAVING_EPOCH
                save_yn = (epoch == 0 or epoch + 1 == cfg.EPOCHS or epoch % cfg.SAVING_EPOCH == 0)

                # Make folder in the saving path for qualified epochs
                if save_yn:
                    self._make_path(epoch)


                for idx in range(cfg.SPLITS):

                    start = time.time()
                    train_X = np.vstack((self.all_X[:idx * self.val_data_length], self.all_X[(idx+1) * self.val_data_length:]))
                    train_Y = np.vstack((self.all_Y[:idx * self.val_data_length], self.all_Y[(idx+1) * self.val_data_length:]))
                    val_X = self.all_X[idx * self.val_data_length : (idx+1) * self.val_data_length]
                    val_Y = self.all_Y[idx * self.val_data_length : (idx+1) * self.val_data_length]

                    # create variables to save results
                    mean_iou_list, unfiltered_iou_list, loss_list = [], [], []
                    total_cost, total_val_iou, total_val_unfiltered_iou, step = 0, 0, 0, 0

                    train_step = train_X.shape[0] // cfg.BATCH_SIZE
                    val_step = val_X.shape[0] // cfg.BATCH_SIZE

                    for batch in tl.iterate.minibatches(inputs=train_X, targets=train_Y,
                                                        batch_size=cfg.BATCH_SIZE, shuffle=True):
                        batch_x, batch_y = batch

                        # make_one_hot
                        key = np.array(cfg.TRAIN_LABEL)
                        _, index = np.unique(batch_y, return_inverse=True)
                        seg = key[index].reshape(batch_y.shape)
                        batch_y = np.eye(4)[seg]


                        # step_time = time.time()
                        tr_feed_dict = {self.model.X: batch_x,
                                        self.model.Y: batch_y,
                                        self.model.training: True,
                                        self.model.drop_rate: 0.2}

                        # cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)
                        pred, Y, cost, _ = sess.run([self.model.pred, self.model.Y, self.model.loss, self.optimizer], feed_dict=tr_feed_dict)

                        pred_list, label_list = utils.convert_to_subregions(pred,
                                                                            Y,
                                                                            [cfg.ET_LABEL, cfg.TC_LABEL, cfg.WT_LABEL],
                                                                            one_hot=True)
                        et_result = utils.cal_result(pred_list[0], label_list[0], one_hot=False)
                        tc_result = utils.cal_result(pred_list[1], label_list[1], one_hot=False)
                        wt_result = utils.cal_result(pred_list[2], label_list[2], one_hot=False)

                        print('-----------------------------')
                        print('et', et_result)
                        print('tc', tc_result)
                        print('wt', wt_result)
                        print('-----------------------------')

                        total_cost += cost
                        step += 1

                        # print out current epoch, step and batch loss value
                        self.result = 'Epoch: {0} / {1}, Cross validation : {2} / {3}, Step: {4} / {5}, Batch loss: {6}'.format((epoch + 1),
                                                                                                                                cfg.EPOCHS,
                                                                                                                                idx + 1,
                                                                                                                                cfg.SPLITS,
                                                                                                                                step,
                                                                                                                                train_step,
                                                                                                                                cost)

                        print(self.result)
                        # utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

                    for batch in tl.iterate.minibatches(inputs=val_X, targets=val_Y,
                                                        batch_size=cfg.BATCH_SIZE, shuffle=True):
                        batch_x, batch_y = batch

                        # make_one_hot
                        key = np.array(cfg.TRAIN_LABEL)
                        _, index = np.unique(batch_y, return_inverse=True)
                        seg = key[index].reshape(batch_y.shape)
                        batch_y = np.eye(4)[seg]



                        val_feed_dict = {self.model.X: batch_x,
                                         self.model.Y: batch_y,
                                         self.model.training: False,
                                         self.model.drop_rate: 0}





                        # Calculate validation Iou(Intersection of Union). Iou is used as an accuracy in image segmentation.
                        # return [acc, mean_iou, unfiltered_iou] in model.iou
                        # val_results, predicted_result, x_list, y_list = sess.run([self.model.results,
                        #                                                           self.model.logit,
                        #                                                           self.model.X,
                        #                                                           self.model.Y],
                        #                                                          feed_dict=val_feed_dict)

                        # # convert received batch iou as a list
                        # ious = list(val_results[0])
                        # accs = list(val_results[1])
                        # unfiltered_iou = np.mean(ious)
                        #
                        # # uses only iou > 0.01 (i.e. IoUs predicting over certain cutline) to calculate IoUs for diagnosis accuracy
                        # iou_list = []
                        #
                        # for iou in ious:
                        #     if iou > 0.01:
                        #         iou_list.append(iou)
                        #
                        # after_filtered_length = len(iou_list)
                        # # before_filtered_length = len(ious)
                        #
                        # # val_batch_acc = after_filtered_length / before_filtered_length
                        #
                        # if after_filtered_length == 0:
                        #     mean_iou = 0
                        #
                        # else:
                        #     mean_iou = np.mean(iou_list)
                        #
                        # # Add IoUs per patch and accuracies to entire IoU value. As epoch terminated, convert to average IoU and ave accuracy.
                        # total_val_iou += mean_iou
                        # total_val_unfiltered_iou += unfiltered_iou



                    end = time.time()
                    training_time = end - start
                    total_training_time += training_time

                    Loss = total_cost / train_step
                    Valdation_IoU = total_val_iou / val_step
                    Valdation_Unfiltered_IoU = total_val_unfiltered_iou / val_step

                    self.result = 'Epoch: {} / {}, Cross validation : {} / {}, Loss: {:.4f}, Validation IoU: {:.4f}, ' \
                                  'Validation Unfiltered IoU: {:.4f}, Training time: {:.2f}'.format((epoch + 1),
                                                                                                    cfg.EPOCHS,
                                                                                                    idx + 1,
                                                                                                    cfg.SPLITS,
                                                                                                    Loss,
                                                                                                    Valdation_IoU,
                                                                                                    Valdation_Unfiltered_IoU,
                                                                                                    training_time)
                    print(self.result)
                    utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

                    result_dict = {self.p_eval.mean_iou: Valdation_IoU,
                                   self.p_eval.tot_iou: Valdation_Unfiltered_IoU,
                                   self.p_eval.loss: Loss}

                    # TB
                    summary = sess.run(self.merged_summary, feed_dict=result_dict)
                    self.writer.add_summary(summary, global_step=epoch)

                    mean_iou_list.append(Valdation_IoU)
                    unfiltered_iou_list.append(Valdation_Unfiltered_IoU)
                    loss_list.append(Loss)

                # 모델 저장
                if save_yn:
                    saver.save(sess, self.model_save_path)
                    print(">>> Model SAVED")
                    print('')

                            # save validation image results
                            # if save_yn:
                            #     self._make_img(predicted_result, x_list, y_list, address, cfg.W, cfg.P)



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

        dir_name = ['merged', 'pred', 'label', 'compare']
        self.path_list = [(self.img_path + '{0}{1}{0}' + name).format(cfg.PATH_SLASH, str(epoch + 1)) for name in dir_name]

if __name__ == "__main__":
    trainer = Train()
    trainer.train()
