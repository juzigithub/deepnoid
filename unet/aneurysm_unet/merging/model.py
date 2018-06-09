'''
model construct

by 신형은 주임
'''
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import utils

from tensorlayer.layers import *


class Model:
    def __init__(self, batch_size, img_size, n_channel, n_class, depth, n_filter):

        self.batch_size = batch_size
        self.img_size = img_size
        self.n_channel = n_channel
        self.n_class = n_class
        self.depth = depth
        self.n_filter = n_filter

        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        self.training = tf.placeholder(tf.bool, name='training')
        self.X = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='X')
        self.Y = tf.placeholder(tf.float32, [None, img_size, img_size, n_class], name='Y')
        self.X_ADD = tf.placeholder(tf.int32, [None, 3], name='X_ADD')



#############################################################
        self.dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y, self.X_ADD)).shuffle(buffer_size=3000)  ### buffer_size 를 cfg로
        self.dataset = self.dataset.batch(self.batch_size).repeat()

        self.iter = self.dataset.make_initializable_iterator()
        self.features, self.labels, self.address = self.iter.get_next()  #### self.X, self.Y 들어갈 자리에 self.features, self.labels 입력
#############################################################




        self.logit = self.u_net()

        self.pred = tf.nn.softmax(logits=self.logit)
        # 활성화 시킨 probability map을 split 하여 foreground와 background로 분리합니다.
        self.foreground_predicted, self.background_predicted = tf.split(self.pred, [1, 1], 3)

        # 라벨이미지 역시 foreground와 background로 분리합니다
        self.foreground_truth, self.background_truth = tf.split(self.labels, [1, 1], 3) ########### self.Y -> self.labels 로
        self.loss = utils.cross_entropy(output=self.foreground_predicted, target=self.foreground_truth)
        self.results = list(utils.iou_coe(output=self.foreground_predicted, target=self.foreground_truth))


    def u_net(self):
        # start down sampling by depth n.

        up_conv = [0] * self.depth
        up_deconv = [0] * self.depth
        up_norm = [0] * self.depth
        up_act = [0] * self.depth
        up_concat = [0] * self.depth
        down_conv = [0] * self.depth
        down_norm = [0] * self.depth
        down_act = [0] * self.depth
        down_pool = [0] * self.depth

        with tf.variable_scope('down'):

            inputs = self.features
            channel_n = self.n_filter
            pool_size = self.img_size

            for i in range(self.depth):

                down_conv[i] = utils.conv2D(str(i) + '_conv1', inputs, channel_n, [3,3], [1,1], padding = 'SAME')
                down_norm[i] = utils.Normalization(down_conv[i], 'batch',self.training, str(i) + '_norm1')
                down_act[i] = utils.activation(str(i) + '_act1', down_norm[i], 'relu')
                down_conv[i] = utils.conv2D(str(i) + '_conv2', down_act[i], channel_n, [3,3], [1,1], 'SAME')
                down_norm[i] = utils.Normalization(down_conv[i], 'batch',self.training, str(i) + '_norm2')
                down_act[i] = utils.activation(str(i) + '_act2', down_norm[i], 'relu')
                down_pool[i] = utils.maxpool(str(i) + '_pool1', down_act[i], [2,2], [2,2], 'SAME')

                inputs = down_pool[i]

                channel_n *= 2
                pool_size //= 2

            down_conv_f = utils.conv2D('final_conv1', inputs, channel_n, [3, 3], [1, 1], padding='SAME')
            down_norm_f = utils.Normalization(down_conv_f, 'batch', self.training, 'final_norm1')
            down_act_f = utils.activation('final_conv1', down_norm_f, 'relu')
            down_conv_f = utils.conv2D('final_conv2', down_act_f, channel_n, [3, 3], [1, 1], padding='SAME')
            down_norm_f= utils.Normalization(down_conv_f, 'batch', self.training, 'final_norm2')
            down_act_f = utils.activation('final_conv2', down_norm_f, 'relu')

        with tf.variable_scope('up'):

            inputs = down_act_f

            for i in reversed(range(self.depth)):

                pool_size *=2

                up_deconv[i] = utils.deconv2D(str(i) +'_upconv1', inputs, [3,3,channel_n //2, channel_n],
                                              [-1, pool_size, pool_size, channel_n // 2], [1, 2, 2, 1], 'SAME')
                up_deconv[i] = tf.reshape(up_deconv[i], shape=[-1, pool_size, pool_size, channel_n // 2])

                up_norm[i] = utils.Normalization(up_deconv[i], 'batch', self.training, str(i) + '_upnorm1')
                up_act[i] = utils.activation(str(i) + '_upact1', up_norm[i], 'relu')
                up_concat[i] = utils.concat(str(i) + '_upconcat1', [up_act[i], down_act[i]], 3)

                channel_n //= 2

                up_conv[i] = utils.conv2D(str(i) + '_upconv1', up_concat[i], channel_n, [3,3], [1,1], 'SAME')
                # up_norm[i] = utils.Normalization(up_conv[i], 'batch', self.training, str(i) + '_upnorm1')
                up_act[i] = utils.activation(str(i) + '_upact1', up_conv[i], 'relu')
                up_conv[i] = utils.conv2D(str(i) + '_upconv2', up_act[i], channel_n, [3,3], [1,1], 'SAME')
                up_norm[i] = utils.Normalization(up_conv[i], 'batch', self.training, str(i) + '_upnorm2')
                up_act[i] = utils.activation(str(i) + '_upact2', up_norm[i], 'relu')

                inputs = up_act[i]

            up_conv_f = utils.conv2D('final_upconv1', inputs, 2, [1,1], [1,1], 'SAME')
            print(up_conv_f)
        return up_conv_f












