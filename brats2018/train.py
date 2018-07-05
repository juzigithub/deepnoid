import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os
import config as cfg
import time
import utils

# import resnet as resnet
# import loadutils as loader
# import file_converter as fc

import brats2018.resnet as resnet
import brats2018.loadutils as loader
import brats2018.file_converter as fc

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class Train:
    def __init__(self):

        '''

        self.data_types = ['flair', 't1', 't1ce', 't2']

        self.splits = 5

        self.train_sets, self.val_sets = loader.cv(cfg.HGG_data_path, self.splits)

        print('self.train_sets.shape : ', self.train_sets.shape)
        print('self.val_sets.shape : ', self.val_sets.shape)

        # self.train_sets_X, self.train_sets_Y, self.val_sets_X, self.val_sets_Y = [], [], [], []
        self.train_sets_X = np.empty((0,26040, 240, 240, 4))
        self.train_sets_Y = np.empty((0,26040, 240, 240))
        self.val_sets_X = np.empty((0,6510,240,240,4))
        self.val_sets_Y = np.empty((0,6510,240,240))

        print('type(self.train_sets_X) : ', type(self.train_sets_X))            # ndarray

        # self.model = resnet.Model()




        for i in range(self.splits):

            self.train_set_X, self.train_set_Y = loader.get_normalized_img(cfg.HGG_data_path, self.train_sets[i], self.data_types)
            self.val_set_X, self.val_set_Y = loader.get_normalized_img(cfg.HGG_data_path, self.val_sets[i], self.data_types)

            # # --------------------- show data --------------------------------#
            #
            # x = self.train_set_X[10]
            # y = self.train_set_Y[10]
            #
            # print('x.shape : ', x.shape)
            # print('y.shape : ', y.shape)
            #
            # vis_imgs(x, y, 'brats2018/{}/_train_im.png')

            print('type(self.train_set_X) : ', type(self.train_set_X))      # list , self.train_X.shape : (26040, 240, 240, 4)
                                                                            # self.train_Y.shpae : Y.shape :  (26040, 240, 240)

            # self.train_sets_X.append(self.train_set_X)
            # self.train_sets_Y.append(self.train_set_Y)
            # self.val_sets_X.append(self.val_set_X)
            # self.val_sets_Y.append(self.val_set_Y)
            self.train_sets_X = np.append(self.train_sets_X, [self.train_set_X], axis=0)
            self.train_sets_Y = np.append(self.train_sets_Y, [self.train_set_Y], axis=0)
            self.val_sets_X = np.append(self.val_sets_X, [self.val_set_X], axis=0)
            self.val_sets_Y = np.append(self.val_sets_Y, [self.val_set_Y], axis=0)

        # self.train_sets_X = np.asarray(self.train_sets_X, dtype=np.float32)          # X.shape :  (26040, 240, 240, 4)
        # self.train_sets_Y = np.asarray(self.train_sets_Y, dtype=np.float32)          # Y.shape :  (26040, 240, 240)
        # self.val_sets_X = np.asarray(self.val_sets_X, dtype=np.float32)          # X.shape :  (26040, 240, 240, 4)
        # self.val_sets_Y = np.asarray(self.val_sets_Y, dtype=np.float32)          # Y.shape :  (26040, 240, 240)

        print('type(self.train_sets_X) : ', type(self.train_sets_X.shape))
        print('self.train_sets_X.shape : ', self.train_sets_X.shape)
        print('self.train_sets_Y.shape : ', self.train_sets_Y.shape)
        print('self.val_sets_X.shape : ', self.val_sets_X.shape)
        print('self.val_sets_Y.shape : ', self.val_sets_Y.shape)

        '''

        # self.data_loader = loader.DataLoader()
        self.model = resnet.Model()
        self.splits = 5

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

        print('')
        print('>>> Data Loading Started')
        print('')

        dstime = time.time()

        # data_list_loader + img_grey_size  -> save as pkl -> pkl data load
        # Unlike other variables composes with valX = [abnorm(0) or norm(1) or else(2), img_da1ta]
        # self.train_sets_X, self.train_sets_Y, self.val_sets_X, self.val_sets_Y = self.data_loader.load_data(type='pkl', mode='train')
        self.train_sets_X = np.load(cfg.HGG_data_path + 'brats_train_image.npy')
        self.train_sets_Y = np.load(cfg.HGG_data_path + 'brats_train_label.npy')
        self.val_sets_X = np.load(cfg.HGG_data_path + 'brats_val_image.npy')
        self.val_sets_Y = np.load(cfg.HGG_data_path + 'brats_val_label.npy')

        print('self.train_sets_X.shape : ', self.train_sets_X.shape)
        print('self.train_sets_Y.shape : ', self.train_sets_Y.shape)
        print('self.val_sets_X.shape : ', self.val_sets_X.shape)
        print('self.val_sets_Y.shape : ', self.val_sets_Y.shape)


        detime = time.time()

        print('')
        print('>>> Data Loading Complete. Consumption Time :', detime - dstime)
        print('')

    def optimizer(self, global_step):
        exponential_decay_learning_rate = tf.train.exponential_decay(learning_rate=cfg.INIT_LEARNING_RATE,
                                                                     global_step=global_step,
                                                                     decay_steps=cfg.DECAY_STEP,
                                                                     decay_rate=cfg.DECAY_RATE,
                                                                     staircase=cfg.DECAY_STAIRCASE,
                                                                     name='learning_rate')

        self.optimizer = utils.select_optimizer(cfg.OPTIMIZER, exponential_decay_learning_rate, self.model.loss, global_step)

    def train(self):

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

            for i in range(self.splits):

                train_X, train_Y = self.train_sets_X[i], self.train_sets_Y[i]
                val_X,val_Y = self.val_sets_X[i], self.val_sets_Y[i]
                print('type(train_X) : ', type(train_X))
                print('type(train_Y) : ', type(train_Y))
                print('train_X.shape : ', train_X.shape)
                print('train_Y.shape : ', train_Y.shape)
                print('-----------------------val------------------------')
                print('type(val_X) : ', type(val_X))
                print('type(val_Y) : ', type(val_Y))
                print('val_X.shape : ', val_X.shape)
                print('val_Y.shape : ', val_Y.shape)
                train_step = len(train_X.shape[0]) // cfg.BATCH_SIZE
                val_step = len(val_X.shape[0]) // cfg.BATCH_SIZE
                print('train_step : ', train_step)
                print('val_step : ', val_step)

                for epoch in range(cfg.EPOCHS):

                    # shuffle

                    # create variables to save results
                    mean_iou_list, unfiltered_iou_list, loss_list = [], [], []
                    total_cost, total_val_iou, total_val_unfiltered_iou, step = 0, 0, 0, 0

                    # save image and model at the first epoch, final epoch and multiples of SAVING_EPOCH
                    save_yn = (epoch == 0 or epoch + 1 == cfg.EPOCHS or epoch % cfg.SAVING_EPOCH == 0)

                    # Make folder in the saving path for qualified epochs
                    if save_yn:
                        self._make_path(epoch)

                    # for bath in range(train_step):
                    for batch in tl.iterate.minibatches(inputs=train_X, targets=train_Y,
                                                        batch_size=cfg.BATCH_SIZE, shuffle=True):

                        batch_x, batch_y = batch
                        step_time = time.time()

                        print('type(batch_x) : ', type(batch_x))
                        print('type(batch_y) : ', type(batch_y))
                        print('batch_x.shape : ', batch_x.shape)
                        print('batch_y.shape : ', batch_y.shape)

                        tr_feed_dict = {self.model.X: batch_x,
                                        self.model.Y: batch_y,
                                        self.model.training: True,
                                        self.model.drop_rate: 0.2}

                        cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)

                        total_cost += cost
                        step += 1

                        # print out current epoch, step and batch loss value
                        # self.result = 'Epoch:', '[%d' % (epoch + 1), '/ %d]  ' % cfg.EPOCHS, 'Step:', step, '/', train_step,'  Batch loss:', cost
                        self.result = 'Epoch: {0} / {1}, Step: {2} / {3}, Batch loss: {4}'.format((epoch + 1),
                                                                                                  cfg.EPOCHS, step,
                                                                                                  train_step, cost)

                        print(self.result)
                        utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

                    for batch in tl.iterate.minibatches(inputs=val_X, targets=val_Y,
                                                        batch_size=cfg.BATCH_SIZE, shuffle=True):

                        batch_x, batch_y = batch
                        step_time = time.time()


                        # feed_dict for iterator
                        val_feed_dict = {self.model.X: batch_x,
                                         self.model.Y: batch_y,
                                         self.model.training: False,
                                         self.model.drop_rate: 0}


                        # Calculate validation Iou(Intersection of Union). Iou is used as an accuracy in image segmentation.
                        # return [acc, mean_iou, unfiltered_iou] in model.iou
                        val_results, predicted_result, x_list, y_list = sess.run([self.model.results,
                                                                                           self.model.foreground_predicted,
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
                        if save_yn:
                            self._make_img(predicted_result, x_list, y_list, address, cfg.W, cfg.P)




    def _make_path(self, epoch):
        ### Savin A Model ###

        # Absolute path for model saving. save as 'file_name'.ckpt
        self.model_save_path = self.model_path + '{0}{1}{0}Unet.ckpt'.format(cfg.PATH_SLASH,
                                                                             str(epoch + 1))

        # create if there is no such file in a saving path
        tl.files.exists_or_mkdir(self.model_path + '{0}{1}'.format(cfg.PATH_SLASH, str(epoch + 1)))

        ### Save validation image result ###

        # val_img_save_path overlaps training image(original image) and predicted image and overlays mask image at affected area
        # raw_val_img_save_path is saving path for predicted image and label_val_img_save_path is for label image
        # save image and model at the first epoch, final epoch and multiples of SAVING_EPOCH

        dir_name = ['merged', 'pred', 'label', 'compare']
        self.path_list = [
            (self.img_path + '{0}{1}{0}' + name).format(cfg.PATH_SLASH, str(epoch + 1)) for name in
            dir_name]

                        # load batch sized x y

if __name__ == "__main__":
    trainer = Train()
    trainer.train()







