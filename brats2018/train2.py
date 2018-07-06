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
# import brats2018.config as cfg
# import brats2018.loader as loader
# from brats2018.model import Model


os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

class Train:
    def __init__(self):

        # self.data_loader = loader.DataLoader()
        # self.model = resnet.Model()
        self.model = Model()
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



        # data_list_loader + img_grey_size  -> save as pkl -> pkl data load
        # Unlike other variables composes with valX = [abnorm(0) or norm(1) or else(2), img_da1ta]
        # self.train_sets_X, self.train_sets_Y, self.val_sets_X, self.val_sets_Y = self.data_loader.load_data(type='pkl', mode='train')
        '''
        self.train_sets_X = np.load(cfg.HGG_data_path + 'brats_train_image.npy')
        self.train_sets_Y = np.load(cfg.HGG_data_path + 'brats_train_label.npy')
        self.val_sets_X = np.load(cfg.HGG_data_path + 'brats_val_image.npy')
        self.val_sets_Y = np.load(cfg.HGG_data_path + 'brats_val_label.npy')

        print('self.train_sets_X.shape : ', self.train_sets_X.shape)
        print('self.train_sets_Y.shape : ', self.train_sets_Y.shape)
        print('self.val_sets_X.shape : ', self.val_sets_X.shape)
        print('self.val_sets_Y.shape : ', self.val_sets_Y.shape)
        '''

        # for i in range(self.splits):
        #
        #     chunk_x = np.load('brats_image_chunk_{}.npy'.format(i))
        #     chunk_y = np.load('brats_label_chunk_{}.npy'.format(i))

        self.all_X = np.concatenate([np.load(cfg.SAVE_DATA_PATH + 'brats_image_chunk_{}.npy'.format(i)) for i in range(cfg.N_FILES)], axis=0)
        self.all_Y = np.concatenate([np.load(cfg.SAVE_DATA_PATH + 'brats_label_chunk_{}.npy'.format(i)) for i in range(cfg.N_FILES)], axis=0)
        self.data_length = np.shape(self.all_X)[0]
        self.val_data_length = self.data_length // 5
        # self.all_X = np.load(cfg.HGG_data_path + 'brats_image.npy') # all_x.shape = [14, 2710, 240, 240, 1] ## 6510 = 155*42
        # self.all_Y = np.load(cfg.HGG_data_path + 'brats_train_label.npy') # all_Y.shape = [2710, 240, 240, 1] ## 6510 = 155*42

        # self.all_X = np.load('brats_image.npy') # all_x.shape = [14, 2710, 240, 240, 1] ## 6510 = 155*42
        # self.all_Y = np.load('brats_label.npy') # all_Y.shape = [2710, 240, 240, 1] ## 6510 = 155*42
        #
        # print('self.all_X.shape : ', self.all_X.shape)
        # print('self.all_Y.shape : ', self.all_Y.shape)


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
                #
                # idx = i*3
                #
                # train_X = np.vstack((self.all_X[:idx], self.all_X[idx+3:]))
                # train_X = np.reshape(train_X, [26040, 240, 240, 4])
                # train_Y = np.vstack((self.all_Y[:idx], self.all_Y[idx+3:]))
                # train_Y = np.reshape(train_Y, [26040, 240, 240, 1])
                # val_X,val_Y = self.all_X[idx:idx+3], self.all_Y[idx:idx+3]
                # train_Y = np.reshape(train_Y, [6510, 240, 240, 1])
                # print('type(train_X) : ', type(train_X))        # <class 'numpy.ndarray'>
                # print('type(train_Y) : ', type(train_Y))        # <class 'numpy.ndarray'>
                # print('train_X.shape : ', train_X.shape)        # (26040, 240, 240, 4)
                # print('train_Y.shape : ', train_Y.shape)        # (26040, 240, 240)
                # print('-----------------------val------------------------')
                # print('type(val_X) : ', type(val_X))        # <class 'numpy.ndarray'>
                # print('type(val_Y) : ', type(val_Y))        # <class 'numpy.ndarray'>
                # print('val_X.shape : ', val_X.shape)        # (6510, 240, 240, 4)
                # print('val_Y.shape : ', val_Y.shape)        # (6510, 240, 240)
                # train_step = train_X.shape[0] // cfg.BATCH_SIZE
                # val_step = val_X.shape[0] // cfg.BATCH_SIZE
                # print('train_step : ', train_step)
                # print('val_step : ', val_step)

                for idx in range(cfg.SPLITS):
                    # train_idx = [i for i in range(15)]
                    # val_idx = [idx + i for i in range(3)]
                    # for i in val_idx:
                    #     train_idx.remove(i)
                    train_X = np.vstack((self.all_X[:idx * self.val_data_length], self.all_X[(idx+1) * self.val_data_length:]))
                    train_Y = np.vstack((self.all_Y[:idx * self.val_data_length], self.all_Y[(idx+1) * self.val_data_length:]))
                    val_X = self.all_X[idx * self.val_data_length : (idx+1) * self.val_data_length]
                    val_Y = self.all_Y[idx * self.val_data_length : (idx+1) * self.val_data_length]
                    #
                    # train_X = np.concatenate([np.load(cfg.SAVE_DATA_PATH + 'brats_image_chunk_{}.npy'.format(i)) for i in train_idx], axis=0)
                    # train_Y = np.concatenate([np.load(cfg.SAVE_DATA_PATH + 'brats_label_chunk_{}.npy'.format(i)) for i in train_idx], axis=0)
                    # val_X = np.concatenate([np.load(cfg.SAVE_DATA_PATH + 'brats_image_chunk_{}.npy'.format(i)) for i in val_idx], axis=0)
                    # val_Y = np.concatenate([np.load(cfg.SAVE_DATA_PATH + 'brats_label_chunk_{}.npy'.format(i)) for i in val_idx], axis=0)
                    # print('train_x_shape : ', np.shape(train_X))
                    # print('train_y_shape : ', np.shape(train_Y))
                    # print('val_x_shape : ', np.shape(val_X))
                    # print('val_y_shape : ', np.shape(val_Y))

                    # create variables to save results
                    mean_iou_list, unfiltered_iou_list, loss_list = [], [], []
                    total_cost, total_val_iou, total_val_unfiltered_iou, step = 0, 0, 0, 0

                    train_step = train_X.shape[0] // cfg.BATCH_SIZE
                    val_step = val_X.shape[0] // cfg.BATCH_SIZE

                    for batch in tl.iterate.minibatches(inputs=train_X, targets=train_Y,
                                                        batch_size=cfg.BATCH_SIZE, shuffle=True):
                        batch_x, batch_y = batch
                        # step_time = time.time()
                        tr_feed_dict = {self.model.X: batch_x,
                                        self.model.Y: batch_y,
                                        self.model.training: True,
                                        self.model.drop_rate: 0.2}

                        cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)

                        total_cost += cost
                        step += 1

                        # print out current epoch, step and batch loss value
                        # self.result = 'Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % cfg.EPOCHS, 'Step:', step, '/', train_step,'  Batch loss:', cost
                        self.result = 'Epoch: {0} / {1}, Cross validation : {2} / {3}, Step: {4} / {5}, Batch loss: {6}'.format((epoch + 1),
                                                                                                                                cfg.EPOCHS,
                                                                                                                                idx + 1,
                                                                                                                                cfg.SPLITS,
                                                                                                                                step,
                                                                                                                                train_step,
                                                                                                                                cost)

                        print(self.result)
                        utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

                    for batch in tl.iterate.minibatches(inputs=val_X, targets=val_Y,
                                                        batch_size=cfg.BATCH_SIZE, shuffle=True):
                        batch_x, batch_y = batch
                        # step_time = time.time()
                        # feed_dict for iterator

                        val_feed_dict = {self.model.X: batch_x,
                                         self.model.Y: batch_y,
                                         self.model.training: False,
                                         self.model.drop_rate: 0}


                        # Calculate validation Iou(Intersection of Union). Iou is used as an accuracy in image segmentation.
                        # return [acc, mean_iou, unfiltered_iou] in model.iou
                        val_results, predicted_result, x_list, y_list = sess.run([self.model.results,
                                                                                  self.model.logit,
                                                                                  self.model.X,
                                                                                  self.model.Y],
                                                                                 feed_dict=val_feed_dict)
                        # acc, val_mean_iou, val_unfiltered_iou = val_results

                        # convert received batch iou as a list
                        ious = list(val_results[0])
                        unfiltered_iou = np.mean(ious)

                        # uses only iou > 0.01 (i.e. IoUs predicting over certain cutline) to calculate IoUs for diagnosis accuracy
                        iou_list = []

                        for iou in ious:
                            if iou > 0.01:
                                iou_list.append(iou)

                        after_filtered_length = len(iou_list)
                        # before_filtered_length = len(ious)

                        # val_batch_acc = after_filtered_length / before_filtered_length

                        if after_filtered_length == 0:
                            mean_iou = 0

                        else:
                            mean_iou = np.mean(iou_list)

                        # Add IoUs per patch and accuracies to entire IoU value. As epoch terminated, convert to average IoU and ave accuracy.
                        total_val_iou += mean_iou
                        total_val_unfiltered_iou += unfiltered_iou

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
