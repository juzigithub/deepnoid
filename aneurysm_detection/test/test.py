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

class Test:
    def __init__(self, restore=True):
        self.model = Model()
        self.p_eval = pe.performance()
        self.restore = restore

        # make paths
        *self.test_start_time, _, _, _, _ = time.localtime()
        self.result_txt = '{}_{}_{}_{}_{}.txt'.format(*self.test_start_time)
        self.model_path = '.{0}model{0}train{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.test_start_time)
        self.ckpt_path = '.{0}best{0}'.format(cfg.PATH_SLASH)
        self.img_path = '.{0}imgs{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.test_start_time)
        self.log_path = '.{0}logs{0}{1}_{2}_{3}_{4}_{5}'.format(cfg.PATH_SLASH,*self.test_start_time)

        with open('.{}config.py'.format(cfg.PATH_SLASH), 'rt') as f:
            tl.files.exists_or_mkdir(self.model_path)
            tl.files.exists_or_mkdir(self.ckpt_path)
            tl.files.exists_or_mkdir(self.img_path)
            self.result = f.read()
            utils.result_saver(self.model_path + cfg.PATH_SLASH + self.result_txt, self.result)

        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_path)

    def test(self):
        # global_step = tf.Variable(0, trainable=False, name='global_step')

        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #     # self.optimizer(global_step)
        #     print("LOAD MODEL")

        with tf.Session() as sess:

            #  Saving a model is saving variables such as weights, ans we call it as ckpt(check point file) in tensorflow
            # It's a tensorflow class saving ckpt file
            saver = tf.train.Saver(max_to_keep=50, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

            self.writer.add_graph(sess.graph)

            # initialize global variables from session. Need to assign initial values for each variables
            sess.run(tf.global_variables_initializer())

            if self.restore:
                saver.restore(sess, self.ckpt_path + 'detector_weights.ckpt')

            print("BEGIN TESTING")

            tot_data_X = np.load(cfg.TEST_NPZ_PATH + 'test_input_ori_{}.npz'.format(cfg.IMG_SIZE[0]))['all']
            tot_data_Y = np.load(cfg.TEST_NPZ_PATH + 'test_label_ori_{}.npz'.format(cfg.IMG_SIZE[0]))['all']
            # len_tot_data = np.shape(tot_data_X)[0]
            # train_X = tot_data_X[:int(len_tot_data * 0.8)]
            # val_X = tot_data_X[int(len_tot_data * 0.8):]
            # train_Y = tot_data_Y[:int(len_tot_data * 0.8)]
            # val_Y = tot_data_Y[int(len_tot_data * 0.8):]


            ### feature_stride 바꿔야 하나.. anchor_scale 과 맞춰야하는지 확인 ######################################################
            anchors = utils.get_anchors(cfg.IMG_SIZE,
                                        cfg.ANCHOR_SCALES,
                                        cfg.ANCHOR_RATIOS,
                                        cfg.FEATURE_STRIDES,
                                        cfg.ANCHOR_STRIDE,
                                        normalization=False)
            # print('anchors', anchors)
            normed_anchors = utils.norm_boxes(anchors, cfg.IMG_SIZE)

            test_step = tot_data_X.shape[0] // cfg.BATCH_SIZE

            print_img_idx = 0

            # save image and model at the first epoch, final epoch and multiples of SAVING_EPOCH

            # test
            for batch_x, batch_y in tl.iterate.minibatches(inputs=deepcopy(tot_data_X), targets=deepcopy(tot_data_Y),
                                                           batch_size=1, shuffle=False):
                batch_x = batch_x[0]
                batch_y = batch_y[0]

                batch_x = batch_x.reshape((-1, 3, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]))
                batch_x = np.transpose(batch_x, (0, 2, 3, 1))

                # if np.ndim(batch_y) == 1:
                #     batch_y = np.expand_dims(batch_y, 0)

                # detector_class_label = batch_y[:,0]
                # detector_bbox_label = deepcopy(batch_y[:,1:])

                # batch_y[:, 1:] = np.round(batch_y[:, 1:] * cfg.IMG_SIZE[0])
                # rpn_class_label = np.expand_dims(batch_y[:,0], -1).reshape((1, -1, 1))
                # gt_boxes = batch_y[:, 1:]
                # rpn_class_label, rpn_bbox_label = utils.build_rpn_targets2(anchors, gt_boxes, cfg)
                # rpn_class_label = np.expand_dims(np.expand_dims(rpn_class_label, 0), -1)
                # rpn_bbox_label = np.expand_dims(rpn_bbox_label, 0)

                val_feed_dict = {self.model.X: batch_x,
                                 self.model.anchors: normed_anchors,
                                 self.model.training: False,
                                 self.model.drop_rate: 0}

                detection_outputs = sess.run([self.model.detection_outputs], feed_dict=val_feed_dict)
                # print('gt_boxes', gt_boxes)
                # print('proposals', np.round(proposals * cfg.IMG_SIZE[0]))
                # print('gt', batch_y)
                # print('detection_outputs', np.round(detection_outputs[:,:4] * cfg.IMG_SIZE[0]), detection_outputs[:,4:])

                # input_img #
                n_batch = len(batch_x)
                input_img = batch_x[n_batch//2, :, :, 1]
                input_img = utils.masking_rgb(input_img, multiply=1)

                # label_img #
                print('detection_outputs_shape', np.array(detection_outputs).shape)
                prob = np.round(np.array(detection_outputs)[:,5], 2)
                bbox = np.round(np.array(detection_outputs)[:,:4] * cfg.IMG_SIZE[0])

                for p, b in zip(prob, bbox):
                    cv2.rectangle(batch_y, (b[1], b[0]), (b[3], b[2]), (255, 255, 255), 1)
                    cv2.putText(batch_y, b, (b[1], b[0] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
                label_img = utils.masking_rgb(batch_y, 'red')

                cv2.imwrite(self.img_path + '/{}.png'.format(print_img_idx) , input_img + label_img)
                print_img_idx += 1
                print('{} / {} img saved'.format(print_img_idx, test_step))

    ##########################################################################
    ######################################
    # input = load_normalize_dcm('d:\\FILE00052.dcm')
    # input = masking_rgb(input, multiply=1)
    # label = load_label('d:\\FILE00052.png')
    # label = masking_rgb(label, 'red')
    # print(input.shape)
    # print(label.shape)
    #
    # cv2.imshow('aa', input)
    # cv2.imshow('a', label)
    # cv2.imshow('aaa', input+label)
    # cv2.waitKey()
    #######################################


    def _make_path(self, epoch):
        # Absolute path for model saving. save as 'file_name'.ckpt
        self.model_save_path = self.model_path + '{0}{1}{0}detector_weights.ckpt'.format(cfg.PATH_SLASH,
                                                                                    str(epoch + 1))

        # create if there is no such file in a saving path
        tl.files.exists_or_mkdir(self.model_path + '{0}{1}'.format(cfg.PATH_SLASH, str(epoch + 1)))


if __name__ == "__main__":
    tester = Test(cfg.RESTORE)
    tester.test()