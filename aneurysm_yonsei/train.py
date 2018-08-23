import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os
import time
import cv2
import loader as loader
import config as cfg
import performance_eval as pe
from model import Model        # choose model
import utils as utils

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

class Train:
    def __init__(self, restore=False):
        self.model = Model()
        self.p_eval = pe.performance()
        self.restore = restore

        if cfg.REBUILD_TRAIN_DATA:
            print('')
            print('>>> Train Data Saving Started')
            print('')

            dstime = time.time()
            tl.files.exists_or_mkdir(cfg.SAVE_TRAIN_DATA_PATH)

            loader.save_resized_dcm_as_npy(cfg.DATA_PATH, cfg.SAVE_TRAIN_DATA_PATH, cfg.TRAIN_FILE_NAME)
            detime = time.time()

            print('')
            print('>>> Task1 Data Saving Complete. Consumption Time :', detime - dstime)
            print('')

        # make paths
        *self.train_start_time, _, _, _, _ = time.localtime()
        self.result_txt = '{}_{}_{}_{}_{}.txt'.format(*self.train_start_time)
        self.model_path = '.{0}model{0}train{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        self.ckpt_path = '.{0}best{0}'.format(cfg.PATH_SLASH)
        self.img_path = '.{0}imgs{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)
        self.log_path = '.{0}logs{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.train_start_time)

        with open('.{}config.py'.format(cfg.PATH_SLASH), 'rt') as f:
            tl.files.exists_or_mkdir(self.model_path)
            tl.files.exists_or_mkdir(self.ckpt_path)
            self.result = f.read()
            utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_path)

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

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer(global_step)

        with tf.Session() as sess:

            #  Saving a model is saving variables such as weights, ans we call it as ckpt(check point file) in tensorflow
            # It's a tensorflow class saving ckpt file
            saver = tf.train.Saver(max_to_keep=50, var_list=tf.global_variables())

            # save graphs from tensorboard
            self.writer.add_graph(sess.graph)

            # initialize global variables from session. Need to assign initial values for each variables
            sess.run(tf.global_variables_initializer())

            if self.restore:
                saver.restore(sess, self.ckpt_path + 'weights.ckpt')

            print("BEGIN TRAINING")
            total_training_time = 0

            whole = np.load(cfg.SAVE_TRAIN_DATA_PATH + 'aneurysm_train.npy')
            train_ratio = int(len(whole) * 0.8)
            train_sets = whole[:train_ratio]
            train_sets = train_sets.reshape((-1, cfg.PATCH_SIZE, cfg.PATCH_SIZE, 2))
            train_sets = np.transpose(train_sets, (3, 0, 1, 2))

            val_sets = whole[train_ratio:]
            val_sets = val_sets.reshape((-1, cfg.PATCH_SIZE, cfg.PATCH_SIZE, 2))
            val_sets = np.transpose(val_sets, (3, 0, 1, 2))

            train_X = np.expand_dims(train_sets[0], axis=-1)
            train_Y = train_sets[1]

            #######################
            nonzero_idx = np.where(train_Y.sum(axis=(1, 2)) != 0.)
            train_X = train_X[nonzero_idx]
            train_Y = train_Y[nonzero_idx]
            #######################

            val_X = np.expand_dims(val_sets[0], axis=-1)
            val_Y = val_sets[1]


            drop_rate = cfg.INIT_DROPOUT_RATE
            loss_ratio = np.array(cfg.LAMBDA)

            for epoch in range(cfg.EPOCHS):
                split_training_time = 0

                # dynamic dropout rate
                drop_rate *= cfg.DROPOUT_INCREASE_RATE

                train_step = train_X.shape[0] // cfg.BATCH_SIZE

                epoch_start = time.time()
                # create variables to save results
                total_cost, step = 0, 0

                # save image and model at the first epoch, final epoch and multiples of SAVING_EPOCH
                save_yn = (epoch == 0 or epoch + 1 == cfg.EPOCHS or epoch % cfg.SAVING_EPOCH == 0)

                if save_yn:
                    # Make folder in the saving path for qualified epochs
                    self._make_path(epoch)

                # train
                for batch in tl.iterate.minibatches(inputs=train_X, targets=train_Y,
                                                    batch_size=cfg.BATCH_SIZE, shuffle=True):
                    batch_x, batch_y = batch

                    # make_one_hot
                    key = np.array([0, 1])
                    _, index = np.unique(batch_y, return_inverse=True)
                    seg = key[index].reshape(batch_y.shape)
                    batch_y = np.eye(2)[seg]

                    tr_feed_dict = {self.model.X: batch_x,
                                    self.model.Y: batch_y,
                                    self.model.training: True,
                                    self.model.loss_ratio: loss_ratio,
                                    self.model.drop_rate: drop_rate}

                    cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)

                    bg, fg = sess.run([self.model.bg_loss, self.model.fg_loss], feed_dict=tr_feed_dict)

                    s = bg + fg
                    print('bg loss ratio : ', (bg/s) * 100)
                    print('fg loss ratio : ', (fg/s) * 100)

                    # Update Loss Ratio for next step
                    loss_ratio = loss_ratio * np.sqrt([bg, fg])
                    loss_ratio = loss_ratio / np.sum(loss_ratio)

                    total_cost += cost
                    step += 1

                    # print out current epoch, step and batch loss value
                    self.result = 'Epoch: {} / {}, ' \
                                  'Step: {} / {}, Batch loss: {}'.format(epoch + 1,
                                                                            cfg.EPOCHS,
                                                                            step,
                                                                            train_step,
                                                                            cost)

                    print(self.result)

                one_epoch_result_list = []

                print_img_idx = 0

                # validation test
                for batch in tl.iterate.minibatches(inputs=val_X, targets=val_Y,
                                                    batch_size=cfg.BATCH_SIZE, shuffle=False):
                    print_img_idx += 1
                    batch_x, batch_y = batch

                    # make_one_hot
                    key = np.array([0, 1])
                    _, index = np.unique(batch_y, return_inverse=True)
                    seg = key[index].reshape(batch_y.shape)
                    batch_y = np.eye(2)[seg]

                    val_feed_dict = {self.model.X: batch_x,
                                     self.model.Y: batch_y,
                                     self.model.training: False,
                                     self.model.drop_rate: 0}

                    pred, label = sess.run([self.model.pred, self.model.Y], feed_dict=val_feed_dict)
                    label_print = np.transpose(label, [-1, 0, 1, 2])

                    pred = np.argmax(pred, axis=-1)
                    label = np.argmax(label, axis=-1)

                    # change label value : [bg, fg] = [0, 1, 2, 4] -> [0, 1, 2, 3]

                    ###############################################
                    _, index = np.unique(pred, return_inverse=True)
                    seg = key[index].reshape(pred.shape)
                    pred_print = np.eye(2)[seg]
                    pred_print = np.transpose(pred_print, [-1, 0, 1, 2])

                    one_batch_result = utils.cal_result2(pred, label, one_hot=False)

                    one_epoch_result_list.append(one_batch_result)

                    ## masking results ###
                    # if save_yn:
                    #     # make img
                    #     for i in range(0, cfg.BATCH_SIZE, cfg.BATCH_SIZE//2):
                    #         pred_mask = utils.masking_rgb(pred_print[1][i], color='red')
                    #         label_mask = utils.masking_rgb(label_print[1][i], color='green')
                    #         ori = np.transpose(batch_x, [-1, 0, 1, 2])
                    #         ori = utils.masking_rgb(ori[0][i], color=None)
                    #
                    #         result_image = 0.7 * (ori + pred_mask)
                    #         compare_image = pred_mask + label_mask
                    #
                    #
                    #         cv2.imwrite('./img/epoch{}/result/batch{}_{}.jpg'.format(epoch+1, print_img_idx, i+1), result_image)
                    #         cv2.imwrite('./img/epoch{}/mask/batch{}_{}_mask.jpg'.format(epoch+1, print_img_idx, i+1), pred_mask)
                    #         cv2.imwrite('./img/epoch{}/mask/batch{}_{}_compare.jpg'.format(epoch+1, print_img_idx, i+1), compare_image)
                    #         cv2.imwrite('./img/epoch{}/original/batch{}_{}.jpg'.format(epoch+1, print_img_idx, i+1), ori)

                one_epoch_mean = np.mean(np.array(one_epoch_result_list), axis=0)

                epoch_end = time.time()
                training_time = epoch_end - epoch_start
                split_training_time += training_time
                total_training_time += training_time

                Loss = total_cost / train_step
                print('one_epoch_mean', one_epoch_mean)

                # print and save result of each epoch
                self.result = '\nEpoch: {} / {}, Loss : {}, Training time: {:.2f}' \
                              '\nResults >>> ' \
                              '\n\t Accuracy : {:.4f}, Sensitivity : {:.4f}, Specificity : {:.4f}' \
                              '\n\t mean_IoU : {:.4f}, Dice_score : {:.4f}, Hausdorff : {:.4f}'.format((epoch + 1),
                                                                                              cfg.EPOCHS,
                                                                                              Loss,
                                                                                              training_time,
                                                                                              *one_epoch_mean)
                print(self.result)
                utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

                result_dict = {self.p_eval.acc: one_epoch_mean[0],
                               self.p_eval.sens: one_epoch_mean[1],
                               self.p_eval.spec: one_epoch_mean[2],
                               self.p_eval.miou: one_epoch_mean[3],
                               self.p_eval.dice: one_epoch_mean[4],
                               self.p_eval.hdorff: one_epoch_mean[5]}

                # TB
                summary = sess.run(self.merged_summary, feed_dict=result_dict)
                self.writer.add_summary(summary, global_step=epoch)

                # save model ckpt
                if save_yn:
                    saver.save(sess, self.model_save_path)
                    print(">>> Model SAVED")
                    print('')


    def _make_path(self, epoch):
        # Absolute path for model saving. save as 'file_name'.ckpt
        self.model_save_path = self.model_path + '{0}{1}{0}weights.ckpt'.format(cfg.PATH_SLASH,
                                                                             str(epoch + 1))

        # create if there is no such file in a saving path
        tl.files.exists_or_mkdir(self.model_path + '{0}{1}'.format(cfg.PATH_SLASH, str(epoch + 1)))
        tl.files.exists_or_mkdir('./img/epoch{}/result/'.format(str(epoch + 1)))
        tl.files.exists_or_mkdir('./img/epoch{}/mask/'.format(str(epoch + 1)))
        if epoch == 0:
            tl.files.exists_or_mkdir('./img/epoch{}/original/'.format(str(epoch + 1)))

if __name__ == "__main__":
    trainer = Train(cfg.RESTORE)
    trainer.train()
