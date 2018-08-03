import numpy as np
import tensorflow as tf
import tensorlayer as tl
import os
import time
import cv2
import random
import loader
import config as cfg
import performance_eval as pe
from model import Model
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

class Train:
    def __init__(self, restore=False):
        self.model = Model()
        self.p_eval = pe.performance()
        self.restore = restore

        if cfg.REBUILD_TASK1_DATA:
            print('')
            print('>>> Task1 Data Saving Started')
            print('')

            dstime = time.time()
            tl.files.exists_or_mkdir(cfg.SAVE_TRAIN_DATA_PATH)

            loader.data_saver([cfg.HGG_DATA_PATH, cfg.LGG_DATA_PATH], cfg.SAVE_TRAIN_DATA_PATH, cfg.SPLITS, train=True)

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

        # TB
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
            saver = tf.train.Saver(max_to_keep=10, var_list=tf.global_variables())

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

            X = np.array(
                [np.load(cfg.SAVE_TRAIN_DATA_PATH + 'brats_image_selected_{}.npy'.format(i)) for i in
                 range(cfg.SPLITS)])
            Y = np.array(
                [np.load(cfg.SAVE_TRAIN_DATA_PATH + 'brats_label_selected_{}.npy'.format(i)) for i in
                 range(cfg.SPLITS)])

            drop_rate = cfg.INIT_DROPOUT_RATE
            loss_ratio = np.array(cfg.LAMBDA)
            for epoch in range(cfg.EPOCHS):
                split_training_time = 0

                et_one_split_result = []
                tc_one_split_result = []
                wt_one_split_result = []

                # random order select validation block
                val_order = [i for i in range(cfg.SPLITS)]
                random.shuffle(val_order)
                # dynamic dropout rate
                drop_rate *= cfg.DROPOUT_INCREASE_RATE

                for idx, val_idx in enumerate(val_order):
                    print('data preparing at epoch {0} step {1}...'.format(epoch+1, idx+1))
                    train_idx = [i for i in range(cfg.SPLITS) if i != val_idx]

                    train_X = np.concatenate([X[i] for i in train_idx], axis=0)
                    train_Y = np.concatenate([Y[i] for i in train_idx], axis=0)
                    val_X = np.load(cfg.SAVE_TRAIN_DATA_PATH + 'brats_image_whole_{}.npy'.format(val_idx))
                    val_Y = np.load(cfg.SAVE_TRAIN_DATA_PATH + 'brats_label_whole_{}.npy'.format(val_idx))
                    val_selected_idx = np.random.randint(len(val_Y), size=int(cfg.VAL_PATCH_RATIO * len(val_Y)))
                    val_X = val_X[val_selected_idx]
                    val_Y = val_Y[val_selected_idx]
                    print('X : {0}, train_X : {1}'.format(X.shape, train_X.shape))
                    print('data preparing complete!')

                    train_step = train_X.shape[0] // cfg.BATCH_SIZE

                    epoch_start = time.time()
                    # create variables to save results
                    total_cost, step = 0, 0

                    # save image and model at the first epoch, final epoch and multiples of SAVING_EPOCH
                    save_yn = (epoch == 0 or epoch + 1 == cfg.EPOCHS or epoch % cfg.SAVING_EPOCH == 0)

                    if save_yn:
                        # Make folder in the saving path for qualified epochs
                        self._make_path(epoch * len(val_order) + idx)

                    # train
                    for batch in tl.iterate.minibatches(inputs=train_X, targets=train_Y,
                                                        batch_size=cfg.BATCH_SIZE, shuffle=True):
                        batch_x, batch_y = batch

                        # make_one_hot
                        key = np.array(cfg.TRAIN_LABEL)
                        _, index = np.unique(batch_y, return_inverse=True)
                        seg = key[index].reshape(batch_y.shape)
                        batch_y = np.eye(4)[seg]

                        tr_feed_dict = {self.model.X: batch_x,
                                        self.model.Y: batch_y,
                                        self.model.training: True,
                                        self.model.drop_rate: drop_rate,
                                        self.model.loss_ratio: loss_ratio}

                        cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)

                        bg, ncr, ed, et = sess.run([self.model.bg_loss, self.model.ncr_loss, self.model.ed_loss, self.model.et_loss],
                                                   feed_dict=tr_feed_dict)

                        s = bg + ncr + ed + et
                        print('bg loss ratio : ', (bg/s) * 100)
                        print('ncr loss ratio : ', ( ncr/s ) * 100)
                        print('ed loss ratio : ', ( ed/s ) * 100)
                        print('et loss ratio : ', ( et/s ) * 100)

                        # Update Loss Ratio for next step
                        loss_ratio = loss_ratio * np.sqrt([bg, ncr, ed, et])
                        loss_ratio = loss_ratio / np.sum(loss_ratio)

                        total_cost += cost
                        step += 1

                        # print out current epoch, step and batch loss value
                        self.result = 'Cross validation : {0} / {1}, Epoch: {2} / {3}, ' \
                                      'Step: {4} / {5}, Batch loss: {6}'.format((idx + 1),
                                                                                cfg.SPLITS,
                                                                                epoch + 1,
                                                                                cfg.EPOCHS,
                                                                                step,
                                                                                train_step,
                                                                                cost)

                        print(self.result)

                    et_one_epoch_result_list = []
                    tc_one_epoch_result_list = []
                    wt_one_epoch_result_list = []
                    print_img_idx = 0

                    # validation test
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

                        # change label value : [bg, ncr, ed, et] = [0, 1, 2, 4] -> [0, 1, 2, 3]
                        _, index = np.unique(pred, return_inverse=True)
                        seg = key[index].reshape(pred.shape)
                        pred_print = np.eye(len(cfg.TRAIN_LABEL))[seg]
                        pred_print = np.transpose(pred_print, [-1, 0, 1, 2])

                        # change label : ncr, ed, et -> et, tc, wt
                        pred_list, label_list = utils.convert_to_subregions(pred, label,
                                                                            [cfg.ET_LABEL, cfg.TC_LABEL, cfg.WT_LABEL],
                                                                            one_hot=False)

                        et_one_batch_result = utils.cal_result(pred_list[0], label_list[0], one_hot=False)
                        tc_one_batch_result = utils.cal_result(pred_list[1], label_list[1], one_hot=False)
                        wt_one_batch_result = utils.cal_result(pred_list[2], label_list[2], one_hot=False)

                        et_one_epoch_result_list.append(et_one_batch_result)
                        tc_one_epoch_result_list.append(tc_one_batch_result)
                        wt_one_epoch_result_list.append(wt_one_batch_result)

                        # if save_yn:
                        #     # make img
                        #     for i in range(0, cfg.BATCH_SIZE, cfg.BATCH_SIZE//2):
                        #         ncr_mask = utils.masking_rgb(pred_print[1][i], color='green')
                        #         ed_mask = utils.masking_rgb(pred_print[2][i], color='blue')
                        #         et_mask = utils.masking_rgb(pred_print[3][i], color='red')
                        #
                        #         et_tc_wt = ed_mask + 2 * ncr_mask + 3 * et_mask
                        #         shape = np.shape(et_tc_wt)
                        #         et_tc_wt_mask = et_tc_wt.reshape([-1,3])
                        #         len_mask = len(et_tc_wt_mask)
                        #         et_tc_wt_mask = et_tc_wt_mask - (0.9*et_tc_wt_mask.max(1).reshape([len_mask, -1]) - et_tc_wt_mask.min(1).reshape([len_mask, -1]))
                        #         et_tc_wt_mask = np.clip(et_tc_wt_mask, 0., 1.) * 255
                        #         et_tc_wt_mask = et_tc_wt_mask.reshape(shape)
                        #
                        #         ori = np.transpose(batch_x, [-1, 0, 1, 2])
                        #         ori = utils.masking_rgb(ori[0][i], color=None)
                        #
                        #         result_image = 0.7 * (ori + et_tc_wt_mask)
                        #
                        #         cv2.imwrite('./img/epoch{}/result/batch{}_{}.jpg'.format(epoch+1, print_img_idx, i+1), result_image)
                        #         cv2.imwrite('./img/epoch{}/mask/batch{}_{}_ncr.jpg'.format(epoch+1, print_img_idx, i+1), ncr_mask)
                        #         cv2.imwrite('./img/epoch{}/mask/batch{}_{}_ed.jpg'.format(epoch+1, print_img_idx, i+1), ed_mask)
                        #         cv2.imwrite('./img/epoch{}/mask/batch{}_{}_et.jpg'.format(epoch+1, print_img_idx, i+1), et_mask)
                        #         cv2.imwrite('./img/epoch{}/mask/batch{}_{}_all.jpg'.format(epoch+1, print_img_idx, i+1), et_tc_wt_mask)
                        #         cv2.imwrite('./img/epoch{}/original/batch{}_{}.jpg'.format(epoch+1, print_img_idx, i+1), ori)

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

                    # print and save result of each epoch
                    self.result = '\nCross validation : {} / {}, Epoch: {} / {}, Loss : {}, Training time: {:.2f}' \
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

                    # save model ckpt
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

                # print and save result of each split
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
                              '#########################################\n'.format((idx + 1),
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

            # print and save total result
            self.result = 'ET >>> ' \
                          '\n\tAccuracy: {:.4f} +- {:.2f} ' \
                          '\n\tSensitivity {:.4f} +- {:.2f} ' \
                          '\n\tSpecificity: {:.4f} +- {:.2f} ' \
                          '\n\tDice Score : {:.4f} +- {:.2f} ' \
                          '\n\tMean IoU : {:.4f} +- {:.2f} ' \
                          '\n\tHausdorff_D : {:.4f} +- {:.2f}'.format(et_total_mean[0], et_total_std[0],
                                                                     et_total_mean[1], et_total_std[1],
                                                                     et_total_mean[2], et_total_std[2],
                                                                     et_total_mean[3], et_total_std[3],
                                                                     et_total_mean[4], et_total_std[4],
                                                                     et_total_mean[5], et_total_std[5])
            utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

            self.result = 'TC >>> ' \
                          '\n\tAccuracy: {:.4f} +- {:.2f} ' \
                          '\n\tSensitivity {:.4f} +- {:.2f} ' \
                          '\n\tSpecificity: {:.4f} +- {:.2f} ' \
                          '\n\tDice Score : {:.4f} +- {:.2f} ' \
                          '\n\tMean IoU : {:.4f} +- {:.2f} ' \
                          '\n\tHausdorff_D : {:.4f} +- {:.2f}'.format(tc_total_mean[0], tc_total_std[0],
                                                                     tc_total_mean[1], tc_total_std[1],
                                                                     tc_total_mean[2], tc_total_std[2],
                                                                     tc_total_mean[3], tc_total_std[3],
                                                                     tc_total_mean[4], tc_total_std[4],
                                                                     tc_total_mean[5], tc_total_std[5])
            utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

            self.result = 'WT >>> ' \
                          '\n\tAccuracy: {:.4f} +- {:.2f} ' \
                          '\n\tSensitivity {:.4f} +- {:.2f} ' \
                          '\n\tSpecificity: {:.4f} +- {:.2f} ' \
                          '\n\tDice Score : {:.4f} +- {:.2f} ' \
                          '\n\tMean IoU : {:.4f} +- {:.2f} ' \
                          '\n\tHausdorff_D : {:.4f} +- {:.2f}'.format(wt_total_mean[0], wt_total_std[0],
                                                                     wt_total_mean[1], wt_total_std[1],
                                                                     wt_total_mean[2], wt_total_std[2],
                                                                     wt_total_mean[3], wt_total_std[3],
                                                                     wt_total_mean[4], wt_total_std[4],
                                                                     wt_total_mean[5], wt_total_std[5])
            utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

    def _make_path(self, epoch):
        # Absolute path for model saving. save as 'file_name'.ckpt
        self.model_save_path = self.model_path + '{0}{1}{0}brats.ckpt'.format(cfg.PATH_SLASH,
                                                                             str(epoch + 1))

        # create if there is no such file in a saving path
        tl.files.exists_or_mkdir(self.model_path + '{0}{1}'.format(cfg.PATH_SLASH, str(epoch + 1)))
        tl.files.exists_or_mkdir('./img/epoch{}/result/'.format(str(epoch + 1)))
        tl.files.exists_or_mkdir('./img/epoch{}/mask/'.format(str(epoch + 1)))
        tl.files.exists_or_mkdir('./img/epoch{}/survival/'.format(str(epoch + 1)))
        if epoch == 0:
            tl.files.exists_or_mkdir('./img/epoch{}/original/'.format(str(epoch + 1)))

if __name__ == "__main__":
    trainer = Train(cfg.RESTORE)
    trainer.train()
