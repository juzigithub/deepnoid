# import cifar10_pickle as cifar10
import tensorflow as tf
import tensorlayer as tl
import os
import time
import numpy as np
import cv2
import config as cfg
import performance_eval as pe
import utils as utils
from model import Model        # choose model
from copy import deepcopy

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
        self.model = Model()
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
            saver2 = tf.train.Saver(max_to_keep=50, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='feature_extractor_pretrain')+
                                                             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rpn_pretrain'))

            # save graphs from tensorboard
            self.writer.add_graph(sess.graph)

            # initialize global variables from session. Need to assign initial values for each variables
            sess.run(tf.global_variables_initializer())

            if self.restore:
                # saver.restore(sess, self.ckpt_path + 'feature_extractor_weights.ckpt') #######################################
                saver2.restore(sess, self.ckpt_path + 'rpn_weights.ckpt')

            print("BEGIN TRAINING")

            drop_rate = cfg.INIT_DROPOUT_RATE

            tot_data_X = np.load(cfg.NPZ_PATH + 'rpn_pretrain_input_ori_{}.npz'.format(cfg.IMG_SIZE[0]))['all']
            tot_data_Y = np.load(cfg.NPZ_PATH + 'rpn_pretrain_label_ori_{}.npz'.format(cfg.IMG_SIZE[0]))['all']
            len_tot_data = np.shape(tot_data_X)[0]
            train_X = tot_data_X[:int(len_tot_data * 0.8)]
            val_X = tot_data_X[int(len_tot_data * 0.8):]
            train_Y = tot_data_Y[:int(len_tot_data * 0.8)]
            val_Y = tot_data_Y[int(len_tot_data * 0.8):]


            ### feature_stride 바꿔야 하나.. anchor_scale 과 맞춰야하는지 확인 ######################################################
            anchors = utils.get_anchors(cfg.IMG_SIZE,
                                        cfg.ANCHOR_SCALES,
                                        cfg.ANCHOR_RATIOS,
                                        cfg.FEATURE_STRIDES,
                                        cfg.ANCHOR_STRIDE,
                                        normalization=False)
            # print('anchors', anchors)
            normed_anchors = utils.norm_boxes(anchors, cfg.IMG_SIZE)

            for epoch in range(cfg.EPOCHS):

                # print('train_Y', train_Y)


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
                for batch_x, batch_y in tl.iterate.minibatches(inputs=deepcopy(train_X), targets=deepcopy(train_Y),
                                                               batch_size=1, shuffle=True):
                    batch_x = batch_x[0]
                    batch_y = batch_y[0]


                    batch_x = batch_x.reshape((-1, 3, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]))
                    batch_x = np.transpose(batch_x, (0, 2, 3, 1))

                    if np.ndim(batch_y) == 1:
                        batch_y = np.expand_dims(batch_y, 0)

                    ###############################
                    gt = deepcopy(batch_y)
                    ###############################

                    batch_y[:,1:] = np.round(batch_y[:,1:] * cfg.IMG_SIZE[0])
                    # rpn_class_label = np.expand_dims(batch_y[:,0], -1).reshape((1, -1, 1))
                    gt_boxes = batch_y[:,1:]
                    rpn_class_label, rpn_bbox_label = utils.build_rpn_targets2(anchors, gt_boxes, cfg)
                    rpn_class_label = np.expand_dims(np.expand_dims(rpn_class_label, 0), -1)
                    rpn_bbox_label = np.expand_dims(rpn_bbox_label, 0)

                    tr_feed_dict = {self.model.X: batch_x,
                                    self.model.anchors: normed_anchors,
                                    self.model.rpn_class_label: rpn_class_label,
                                    self.model.rpn_bbox_label: rpn_bbox_label,
                                    self.model.training: True,
                                    self.model.drop_rate: drop_rate}

                    cost, _, proposals = sess.run([self.model.loss, self.optimizer, self.model.proposals], feed_dict=tr_feed_dict)
                    print('gt', gt)
                    print('proposals', proposals)
                    print(cost)


                    # Update Loss Ratio for next step
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

                # print_img_idx = 0




###################################################################
                # validation test
                for batch_x, batch_y in tl.iterate.minibatches(inputs=deepcopy(val_X), targets=deepcopy(val_Y),
                                                    batch_size=cfg.BATCH_SIZE, shuffle=False):
                    batch_x = batch_x[0]
                    batch_y = batch_y[0]

                    batch_x = batch_x.reshape((-1, 3, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]))
                    batch_x = np.transpose(batch_x, (0, 2, 3, 1))

                    if np.ndim(batch_y) == 1:
                        batch_y = np.expand_dims(batch_y, 0)
                    batch_y[:, 1:] = np.round(batch_y[:, 1:] * cfg.IMG_SIZE[0])
                    # rpn_class_label = np.expand_dims(batch_y[:,0], -1).reshape((1, -1, 1))
                    gt_boxes = batch_y[:, 1:]
                    rpn_class_label, rpn_bbox_label = utils.build_rpn_targets2(anchors, gt_boxes, cfg)
                    rpn_class_label = np.expand_dims(np.expand_dims(rpn_class_label, 0), -1)
                    rpn_bbox_label = np.expand_dims(rpn_bbox_label, 0)

                    val_feed_dict = {self.model.X: batch_x,
                                     self.model.anchors: normed_anchors,
                                     self.model.rpn_class_label: rpn_class_label,
                                     self.model.rpn_bbox_label: rpn_bbox_label,
                                     self.model.training: False,
                                     self.model.drop_rate: 0}

                    cost, proposals = sess.run([self.model.loss, self.model.proposals], feed_dict=val_feed_dict)
                    print('gt_boxes', gt_boxes)
                    print('proposals', np.round(proposals * cfg.IMG_SIZE[0]))
                    one_epoch_result_list.append(cost)


                one_epoch_mean = np.mean(np.array(one_epoch_result_list))
                self.result = '\nEpoch: {} / {}, Loss : {}\n'.format(epoch + 1, cfg.EPOCHS, one_epoch_mean)
                print(self.result)
                utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

                # save model ckpt
                if save_yn:
                    saver2.save(sess, self.model_save_path)
                    print(">>> Model SAVED")
                    print('')
##########################################################################

    def _make_path(self, epoch):
        # Absolute path for model saving. save as 'file_name'.ckpt
        self.model_save_path = self.model_path + '{0}{1}{0}rpn_weights.ckpt'.format(cfg.PATH_SLASH,
                                                                                    str(epoch + 1))

        # create if there is no such file in a saving path
        tl.files.exists_or_mkdir(self.model_path + '{0}{1}'.format(cfg.PATH_SLASH, str(epoch + 1)))


if __name__ == "__main__":
    trainer = Train(cfg.RESTORE)
    trainer.train()
