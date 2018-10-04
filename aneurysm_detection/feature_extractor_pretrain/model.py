import tensorflow as tf
import utils as utils
import config as cfg

class Model:
    def __init__(self):
        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        self.training = tf.placeholder(tf.bool, name='training')
        self.X = tf.placeholder(tf.float32, [None, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1], 3], name='X')

        self.logit, self.mean, self.gamma = self.model()

        # tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
        # self.reconstruction_loss = tf.losses.mean_squared_error(utils.flatten('X_flatten', self.X), self.logit)

        self.reconstruction_loss = tf.reduce_sum(tf.squared_difference(utils.flatten('logit_flatten', tf.sigmoid(self.logit)),
                                                                       utils.flatten('X_flatten', tf.sigmoid(self.X))),
                                                 1)
        self.latent_loss = 0.5 * tf.reduce_sum(tf.exp(self.gamma) + tf.square(self.mean) - 1 - self.gamma, 1)

        # self.reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits



        # self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        self.loss = tf.reduce_mean(self.reconstruction_loss + self.latent_loss)

    # def feature_extractor(self, inputs, channel_n, n_layer):
    #     with tf.variable_scope('feature_extractor_pretrain'):
    #         l = inputs
    #         for idx in range(n_layer):
    #             l = utils.residual_block_dw_dr(name='downconv_{}'.format(idx),
    #                                            inputs=l,
    #                                            channel_n=channel_n,
    #                                            width_mul=1.0,
    #                                            group_n=cfg.GROUP_N,
    #                                            drop_rate=self.drop_rate,
    #                                            act_fn=cfg.ACTIVATION_FUNC,
    #                                            norm_type=cfg.NORMALIZATION_TYPE,
    #                                            training=self.training,
    #                                            idx=idx)
    #             if idx + 1 < n_layer:
    #                 l = utils.maxpool(name='maxpool_{}'.format(idx),
    #                                   inputs=l,
    #                                   pool_size=[2, 2],
    #                                   strides=[2, 2],
    #                                   padding='same')
    #                 channel_n *= 2
    #             print(l)
    #     return l

    def feature_extractor(self, inputs, channel_n, n_layer):
        with tf.variable_scope('feature_extractor_pretrain'):
            l = inputs
            for idx in range(n_layer):
                l = utils.residual_block_dw_dr(name='downconv_{}'.format(idx),
                                               inputs=l,
                                               channel_n=channel_n,
                                               width_mul=1.0,
                                               group_n=cfg.GROUP_N,
                                               drop_rate=self.drop_rate,
                                               act_fn=cfg.ACTIVATION_FUNC,
                                               norm_type=cfg.NORMALIZATION_TYPE,
                                               training=self.training,
                                               idx=idx)
                channel_n *= 2

                if idx + 1 <= cfg.N_DOWNSAMPLING:
                    l = utils.maxpool(name='maxpool_{}'.format(idx),
                                      inputs=l,
                                      pool_size=[2, 2],
                                      strides=[2, 2],
                                      padding='same')
                print(l)
        return l

    # def reconstructor(self, inputs, output_channel, n_layer):
    #     with tf.variable_scope('non_pretrain'):
    #         l = inputs
    #         # _, h, w, channel_n = tf.shape(l)
    #         _, h, w, channel_n = l.get_shape().as_list()
    #
    #         for idx in range(n_layer):
    #             l = utils.residual_block_dw_dr(name='upconv_{}'.format(idx),
    #                                            inputs=l,
    #                                            channel_n=channel_n,
    #                                            width_mul=1.0,
    #                                            group_n=cfg.GROUP_N,
    #                                            drop_rate=self.drop_rate,
    #                                            act_fn=cfg.ACTIVATION_FUNC,
    #                                            norm_type=cfg.NORMALIZATION_TYPE,
    #                                            training=self.training,
    #                                            idx=idx)
    #             if idx + 1 < n_layer:
    #                 h *= 2
    #                 w *= 2
    #                 channel_n //= 2
    #                 l = utils.select_upsampling(name='upsampling_{}'.format(idx),
    #                                              up_conv=l,
    #                                              up_pool=[],
    #                                              channel_n=channel_n,
    #                                              pool_size_h=h,
    #                                              pool_size_w=w,
    #                                              mode=cfg.UPSAMPLING_TYPE)
    #             print(l)
    #
    #         l = utils.conv2D('outconv', l, output_channel, [1, 1], [1, 1], 'SAME')
    #         l = utils.Normalization(l, cfg.NORMALIZATION_TYPE, self.training, 'outconv_norm', G=cfg.GROUP_N)
    #         l = utils.activation('outconv_act', l, cfg.ACTIVATION_FUNC)
    #         print(l)
    #     return l

    def reconstructor(self, inputs, output_channel, n_layer):
        with tf.variable_scope('non_pretrain'):
            l = inputs
            # _, h, w, channel_n = tf.shape(l)
            _, h, w, channel_n = l.get_shape().as_list()

            for idx in range(n_layer):
                channel_n //= 2

                l = utils.residual_block_dw_dr(name='upconv_{}'.format(idx),
                                               inputs=l,
                                               channel_n=channel_n,
                                               width_mul=1.0,
                                               group_n=cfg.GROUP_N,
                                               drop_rate=self.drop_rate,
                                               act_fn=cfg.ACTIVATION_FUNC,
                                               norm_type=cfg.NORMALIZATION_TYPE,
                                               training=self.training,
                                               idx=idx)
                if idx + 1 <= cfg.N_DOWNSAMPLING:
                    h *= 2
                    w *= 2
                    l = utils.select_upsampling(name='upsampling_{}'.format(idx),
                                                 up_conv=l,
                                                 up_pool=[],
                                                 channel_n=channel_n,
                                                 pool_size_h=h,
                                                 pool_size_w=w,
                                                 mode=cfg.UPSAMPLING_TYPE)
                print(l)

            l = utils.conv2D('outconv', l, output_channel, [1, 1], [1, 1], 'SAME')
            l = utils.Normalization(l, cfg.NORMALIZATION_TYPE, self.training, 'outconv_norm', G=cfg.GROUP_N)
            l = utils.activation('outconv_act', l, cfg.ACTIVATION_FUNC)
            print(l)
        return l


    def model(self):

        inputs = tf.identity(self.X)
        channel_n = cfg.INIT_N_FILTER
        inputs = self.feature_extractor(inputs, channel_n, cfg.PRETRAIN_N_LAYERS)

        # inputs_shape = tf.shape(inputs)
        inputs_shape = inputs.get_shape().as_list()

        print(inputs_shape)
        reshaped_dim = [-1, inputs_shape[1], inputs_shape[2], inputs_shape[3]]

        inputs = utils.flatten('flatten1', inputs)

        mean = utils.fully_connected('mean', inputs, 30)
        gamma = utils.fully_connected('gamma', inputs, 30)
        noise = tf.random_normal(tf.shape(gamma), dtype=tf.float32)
        inputs = mean + tf.exp(0.5 * gamma) * noise

        # inputs = tf.layers.dense(inputs, inputs_shape[1]*inputs_shape[2]*inputs_shape[3]/2, activation=tf.nn.elu)
        inputs = tf.layers.dense(inputs, inputs_shape[1]*inputs_shape[2]*inputs_shape[3], activation=tf.nn.elu)
        inputs = tf.reshape(inputs, reshaped_dim)

        outputs = self.reconstructor(inputs, 3, cfg.PRETRAIN_N_LAYERS)


        # outputs = utils.flatten('flatten2', inputs)
        # outputs = tf.sigmoid(inputs)

        # with tf.variable_scope('new'):
        #     outputs = utils.fully_connected('fc3', outputs, 10)
        return outputs, mean, gamma