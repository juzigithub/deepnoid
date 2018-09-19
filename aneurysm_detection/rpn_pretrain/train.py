# import cifar10_pickle as cifar10
import utils as utils
import tensorflow as tf
import tensorlayer as tl
import os
import time
import numpy as np
import cv2
import config as cfg
import performance_eval as pe
# from model import Model        # choose model

# tot_data, tot_label = cifar10.load_training_data()


# print(np.shape(tot_data))
# # new_data = np.zeros(shape=(10000, 128, 128, 3))
# new_data = np.zeros(shape=(10000,))
# i = 0
# for idx, img in enumerate(tot_label):
#     # new_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
#     # print(idx, new_img.shape)
#     new_data[idx - 10000 * i] = img
#     if (idx + 1) % 10000 == 0:
#         np.save('d:\\cifar10_128_label_{}.npy'.format( (idx + 1) // 10000), new_data)
#         new_data = np.zeros(shape=(10000,))
#         i += 1


os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU

class Train:
    def __init__(self, restore=False):
        # self.model = Model()
        self.p_eval = pe.performance()
        self.restore = restore

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
            tl.files.exists_or_mkdir(self.img_path)
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
            saver = tf.train.Saver(max_to_keep=50, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='feature_extractor_pretrain'))

            # save graphs from tensorboard
            self.writer.add_graph(sess.graph)

            # initialize global variables from session. Need to assign initial values for each variables
            sess.run(tf.global_variables_initializer())

            if self.restore:
                saver.restore(sess, self.ckpt_path + 'feature_extractor_weights.ckpt')

            print("BEGIN TRAINING")
            total_training_time = 0

            # tot_data, tot_label = cifar10.load_training_data()

            # tot_data_shape = np.shape(tot_data)
            # tot_data = tot_data.reshape(len(tot_data), -1)
            # tot_data = scale(tot_data)
            # tot_data = (tot_data - np.mean(tot_data, axis=1)) / np.std(tot_data, axis=1)
            # tot_data = tot_data.reshape(tot_data_shape)
            drop_rate = cfg.INIT_DROPOUT_RATE




            tot_data_X = np.load(cfg.NPZ_PATH + 'rpn_pretrain_input_ori_{}.npz'.format(cfg.IMG_SIZE[0]))['all']
            tot_data_Y = np.load(cfg.NPZ_PATH + 'rpn_pretrain_label_ori_{}.npz'.format(cfg.IMG_SIZE[0]))['all']
            len_tot_data = np.shape(tot_data_X)[0]
            train_X = tot_data_X[:int(len_tot_data * 0.8)]
            val_X = tot_data_X[int(len_tot_data * 0.8):]
            train_Y = tot_data_Y[:int(len_tot_data * 0.8)]
            val_Y = tot_data_Y[int(len_tot_data * 0.8):]

            anchors = utils.get_anchors(cfg.IMG_SIZE,
                                        cfg.ANCHOR_SCALES,
                                        cfg.ANCHOR_RATIOS,
                                        cfg.FEATURE_STRIDES,
                                        cfg.ANCHOR_STRIDE)

            print('len', len_tot_data)
            print('anchors', anchors)













            for epoch in range(cfg.EPOCHS):




                # dynamic dropout rate

                drop_rate *= cfg.DROPOUT_INCREASE_RATE

                train_step = train_X.shape[0] // cfg.BATCH_SIZE

                # epoch_start = time.time()
                # create variables to save results
                total_cost, step = 0, 0

                # save image and model at the first epoch, final epoch and multiples of SAVING_EPOCH
                save_yn = (epoch == 0 or epoch + 1 == cfg.EPOCHS or epoch % cfg.SAVING_EPOCH == 0)

                if save_yn:
                    # Make folder in the saving path for qualified epochs
                    self._make_path(epoch)

                # train
                for batch_x, batch_y in tl.iterate.minibatches(inputs=train_X, targets=train_Y,
                                                               batch_size=cfg.BATCH_SIZE, shuffle=True):
                    # def build_rpn_targets(self, anchors, gt_boxes, config):
                    #     """Given the anchors and GT boxes, compute overlaps and identify positive
                    #     anchors and deltas to refine them to match their corresponding GT boxes.
                    #
                    #     오버랩 계산/positive anchors 계산
                    #     rpn_match : anchor들과 gt_box를 비교하여 positive/negative 판별
                    #     rpn_bbox : delta : anchor box를 gt_box에 맞추기 위한 보정치(delta) 계산
                    #
                    #     anchors: [num_anchors, (y1, x1, y2, x2)]
                    #     gt_class_ids: [num_gt_boxes] Integer class IDs.
                    #     gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
                    #
                    #     Returns:
                    #     rpn_match: [N] (int32) matches between anchors and GT boxes.
                    #                1 = positive anchor, -1 = negative anchor, 0 = neutral
                    #     rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
                    #     """

                    # self.rpn_class_label = tf.placeholder(tf.float32, [cfg.BATCH_SIZE, None, 1], name='rpn_class_label')
                    # self.rpn_bbox_label = tf.placeholder(tf.float32, [cfg.BATCH_SIZE, None, 4], name='rpn_bbox_label')



                    if np.ndim(batch_y) == 1:
                        batch_y = np.expand_dims(batch_y, 0)
                    rpn_class_label = np.expand_dims(batch_y[:,0], -1).reshape((cfg.BATCH_SIZE, -1, 1))
                    rpn_bbox_label = batch_y[:,1:].reshape((cfg.BATCH_SIZE, -1, 1))

                    print('class_label',rpn_class_label.shape)
                    print('bbox_label',rpn_bbox_label.shape)


            #         # batch_x = np.expand_dims(batch_x, axis=-1)
            #         # batch_x = np.concatenate((0.8 * batch_x, batch_x, 1.2 * batch_x), axis=-1)
            #
            #
            #         # # make_one_hot
            #         # key = np.array([0, 1])
            #         # _, index = np.unique(batch_y, return_inverse=True)
            #         # seg = key[index].reshape(batch_y.shape)
            #         # batch_y = np.eye(2)[seg]
            #
            #         tr_feed_dict = {self.model.X: batch_x,
            #                         self.model.training: True,
            #                         self.model.drop_rate: drop_rate}
            #
            #         cost, _ = sess.run([self.model.loss, self.optimizer], feed_dict=tr_feed_dict)
            #
            #         print(cost)
            #
            #         # Update Loss Ratio for next step
            #
            #         total_cost += cost
            #         step += 1
            #
            #         # print out current epoch, step and batch loss value
            #         self.result = 'Epoch: {} / {}, ' \
            #                       'Step: {} / {}, Batch loss: {}'.format(epoch + 1,
            #                                                                 cfg.EPOCHS,
            #                                                                 step,
            #                                                                 train_step,
            #                                                                 cost)
            #
            #         print(self.result)
            #
            #     one_epoch_result_list = []
            #
            #     print_img_idx = 0
            #
            #     # validation test
            #     for batch in tl.iterate.minibatches(inputs=val_X, targets=val_Y,
            #                                         batch_size=cfg.BATCH_SIZE, shuffle=False):
            #         print_img_idx += 1
            #         batch_x, _ = batch
            #         batch_x = np.expand_dims(batch_x, axis=-1)
            #         batch_x = np.concatenate((0.8 * batch_x, batch_x, 1.2 * batch_x), axis=-1)
            #
            #         # # make_one_hot
            #         # key = np.array([0, 1])
            #         # _, index = np.unique(batch_y, return_inverse=True)
            #         # seg = key[index].reshape(batch_y.shape)
            #         # batch_y = np.eye(2)[seg]
            #
            #         val_feed_dict = {self.model.X: batch_x,
            #                          self.model.training: False,
            #                          self.model.drop_rate: 0}
            #
            #         loss, logit = sess.run([self.model.loss, self.model.logit], feed_dict=val_feed_dict)
            #
            #         one_epoch_result_list.append(loss)
            #
            #
            #         # if epoch % 5 == 0 :
            #         #     logit = np.reshape(logit, (-1, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3))
            #         #
            #         #
            #         #     cv2.imwrite(self.img_path + '/{}_{}_{}_original.png'.format(epoch, i, print_img_idx), utils.masking_rgb(batch_x[0,:,:,1]))
            #         #     cv2.imwrite(self.img_path + '/{}_{}_{}_reconstruction.png'.format(epoch, i, print_img_idx), utils.masking_rgb(logit[0,:,:,1]))
            #
            #     one_epoch_mean = np.mean(np.array(one_epoch_result_list))
            #     self.result = '\nEpoch: {} / {}, Loss : {}\n'.format(epoch, cfg.EPOCHS, one_epoch_mean)
            #     print(self.result)
            #     utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)
            #
            #         # label_print = np.transpose(label, [-1, 0, 1, 2])
            #         #
            #         # pred = np.argmax(pred, axis=-1)
            #         # label = np.argmax(label, axis=-1)
            #
            #         # change label value : [bg, fg] = [0, 1, 2, 4] -> [0, 1, 2, 3]
            #
            #         ###############################################
            #     #     _, index = np.unique(pred, return_inverse=True)
            #     #     seg = key[index].reshape(pred.shape)
            #     #     pred_print = np.eye(2)[seg]
            #     #     pred_print = np.transpose(pred_print, [-1, 0, 1, 2])
            #     #
            #     #     one_batch_result = utils.cal_result3(pred, label, one_hot=False)
            #     #
            #     #     one_epoch_result_list.append(one_batch_result)
            #     #
            #     #     ### masking results ###
            #     #     if save_yn:
            #     #         # make img
            #     #         for i in range(0, cfg.BATCH_SIZE):
            #     #             pred_mask = utils.masking_rgb(pred_print[1][i], color='red')
            #     #             label_mask = utils.masking_rgb(label_print[1][i], color='green')
            #     #             ori = np.transpose(batch_x, [-1, 0, 1, 2])
            #     #             ori = ori / np.max(ori)
            #     #             ori = utils.masking_rgb(ori[0][i], color=None)
            #     #
            #     #             result_image = 1.0 * (ori + pred_mask)
            #     #             compare_image = pred_mask + label_mask
            #     #
            #     #
            #     #             cv2.imwrite('./img/epoch{}/result/batch{}_{}.jpg'.format(epoch+1, print_img_idx, i+1), result_image)
            #     #             # cv2.imwrite('./img/epoch{}/mask/batch{}_{}_mask.jpg'.format(epoch+1, print_img_idx, i+1), pred_mask)
            #     #             cv2.imwrite('./img/epoch{}/mask/batch{}_{}_compare.jpg'.format(epoch+1, print_img_idx, i+1), compare_image)
            #     #             if epoch == 0 :
            #     #                 cv2.imwrite('./img/epoch{}/original/batch{}_{}.jpg'.format(epoch+1, print_img_idx, i+1), ori)
            #     #
            #     # one_epoch_mean = np.mean(np.array(one_epoch_result_list), axis=0)
            #     #
            #     # epoch_end = time.time()
            #     # training_time = epoch_end - epoch_start
            #     # split_training_time += training_time
            #     # total_training_time += training_time
            #     #
            #     # Loss = total_cost / train_step
            #     # print('one_epoch_mean', one_epoch_mean)
            #     # #
            #     # # print and save result of each epoch
            #     # self.result = '\nEpoch: {} / {}, Loss : {}, Training time: {:.2f}' \
            #     #               '\nResults >>> ' \
            #     #               '\n\t Accuracy : {:.4f}, Sensitivity : {:.4f}, Specificity : {:.4f}' \
            #     #               '\n\t mean_IoU : {:.4f}, Dice_score : {:.4f}, Hausdorff : {:.4f}'.format((epoch + 1),
            #     #                                                                               cfg.EPOCHS,
            #     #                                                                               Loss,
            #     #                                                                               training_time,
            #     #                                                                               *one_epoch_mean)
            #     # print(self.result)
            #     # utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)
            #     #
            #     # result_dict = {self.p_eval.acc: one_epoch_mean[0],
            #     #                self.p_eval.sens: one_epoch_mean[1],
            #     #                self.p_eval.spec: one_epoch_mean[2],
            #     #                self.p_eval.miou: one_epoch_mean[3],
            #     #                self.p_eval.dice: one_epoch_mean[4],
            #     #                self.p_eval.hdorff: one_epoch_mean[5]}
            #
            #     # TB
            #     # summary = sess.run(self.merged_summary, feed_dict=result_dict)
            #     # self.writer.add_summary(summary, global_step=epoch)
            #
            #     # save model ckpt
            #     if save_yn:
            #         saver.save(sess, self.model_save_path)
            #         print(">>> Model SAVED")
            #         print('')


    def _make_path(self, epoch):
        # Absolute path for model saving. save as 'file_name'.ckpt
        self.model_save_path = self.model_path + '{0}{1}{0}feature_extractor_weights.ckpt'.format(cfg.PATH_SLASH,
                                                                             str(epoch + 1))

        # create if there is no such file in a saving path
        tl.files.exists_or_mkdir(self.model_path + '{0}{1}'.format(cfg.PATH_SLASH, str(epoch + 1)))
        # tl.files.exists_or_mkdir('./img/epoch{}/result/'.format(str(epoch + 1)))
        # tl.files.exists_or_mkdir('./img/epoch{}/mask/'.format(str(epoch + 1)))
        # if epoch == 0:
        #     tl.files.exists_or_mkdir('./img/epoch{}/original/'.format(str(epoch + 1)))

if __name__ == "__main__":
    trainer = Train(cfg.RESTORE)
    trainer.train()
