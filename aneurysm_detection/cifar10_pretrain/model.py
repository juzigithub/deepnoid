import tensorflow as tf
import aneurysm_detection.cifar10_pretrain.utils as utils
import aneurysm_detection.cifar10_pretrain.config as cfg

class Model:
    def __init__(self):
        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        self.training = tf.placeholder(tf.bool, name='training')
        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3], name='X')
        self.Y = tf.placeholder(tf.int64, [None, ], name='Y')

        self.logit = self.model()
        self.label = tf.one_hot(self.Y, 10, dtype=tf.int32)
        print('label_shape', self.label)
        self.classify_loss = utils.cross_entropy(output=self.logit, target=self.label)
        self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([self.classify_loss] + self.reg_loss)
        # self.accuracy = tf.contrib.metrics.accuracy(predictions=tf.cast(tf.nn.softmax(self.logit), tf.int32), labels=self.label)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logit, 1), self.Y), dtype=tf.float32),
                                       name='accuracy')
    def model(self):
        with tf.variable_scope('model'):

            inputs = self.X
            n_channel = cfg.INIT_N_FILTER
            # def residual_block_dw_dr(name, inputs, channel_n, width_mul, group_n, drop_rate, act_fn, norm_type,
            #                          training, idx, rate=None, shortcut=True):

            inputs = utils.residual_block_dw_dr(name='rsconv_0',
                                                inputs=inputs,
                                                channel_n=n_channel,
                                                width_mul=1.0,
                                                group_n=cfg.GROUP_N,
                                                drop_rate=self.drop_rate,
                                                act_fn=cfg.ACTIVATION_FUNC,
                                                norm_type=cfg.NORMALIZATION_TYPE,
                                                training=self.training,
                                                idx = 0)
            print(inputs)

            inputs = utils.maxpool(name='maxpool_0',
                                   inputs=inputs,
                                   pool_size=[2,2],
                                   strides=[2,2],
                                   padding='same')
            print(inputs)
            n_channel *= 2
            inputs = utils.residual_block_dw_dr(name='rsconv_1',
                                                inputs=inputs,
                                                channel_n=n_channel,
                                                width_mul=1.0,
                                                group_n=cfg.GROUP_N,
                                                drop_rate=self.drop_rate,
                                                act_fn=cfg.ACTIVATION_FUNC,
                                                norm_type=cfg.NORMALIZATION_TYPE,
                                                training=self.training,
                                                idx = 0)
            print(inputs)

            inputs = utils.maxpool(name='maxpool_1',
                                   inputs=inputs,
                                   pool_size=[2, 2],
                                   strides=[2, 2],
                                   padding='same')
            print(inputs)
            n_channel *= 2
            inputs = utils.residual_block_dw_dr(name='rsconv_2',
                                                inputs=inputs,
                                                channel_n=n_channel,
                                                width_mul=1.0,
                                                group_n=cfg.GROUP_N,
                                                drop_rate=self.drop_rate,
                                                act_fn=cfg.ACTIVATION_FUNC,
                                                norm_type=cfg.NORMALIZATION_TYPE,
                                                training=self.training,
                                                idx = 0)
            print(inputs)

            # def atrous_spatial_pyramid_pooling(name, inputs, channel_n, atrous_rate_list, act_fn, training):

            inputs = utils.atrous_spatial_pyramid_pooling(name='aspp',
                                                          inputs=inputs,
                                                          channel_n=n_channel,
                                                          atrous_rate_list=[[2,2],[3,3],[4,4]],
                                                          act_fn=cfg.ACTIVATION_FUNC,
                                                          training=self.training)
            print(inputs)
            outputs = utils.flatten('flatten', inputs)
            print(outputs)
            outputs = utils.fully_connected('fc1', outputs, 100)
            print(outputs)
            outputs = utils.fully_connected('fc2', outputs, 10)
            print(outputs)

        # with tf.variable_scope('new'):
        #     outputs = utils.fully_connected('fc3', outputs, 10)
        return outputs