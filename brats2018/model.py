import tensorflow as tf

import utils
import config as cfg
# import brats2018.utils as utils
# import brats2018.config as cfg


class Model:
    def __init__(self):
        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        self.training = tf.placeholder(tf.bool, name='training')
        self.X = tf.placeholder(tf.float32, [None, cfg.IMG_SIZE, cfg.IMG_SIZE, 4], name='X')
        self.Y = tf.placeholder(tf.float32, [None, cfg.IMG_SIZE, cfg.IMG_SIZE, 4], name='Y')
        self.logit = self.resnet()

        self.pred = tf.nn.softmax(logits=self.logit)

        self.bg_pred, self.ncr_pred, self.ed_pred, self.et_pred = tf.split(self.pred, [1,1,1,1], axis=3)
        self.bg_label, self.ncr_label, self.ed_label, self.et_label = tf.split(self.Y, [1,1,1,1], axis=3)

        self.bg_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.bg_pred, target=self.bg_label)
        self.ncr_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.ncr_pred, target=self.ncr_label)
        self.ed_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.ed_pred, target=self.bg_label)
        self.et_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.bg_pred, target=self.bg_label)


        # self.loss = cfg.LAMBDA[0] * self.bg_loss + cfg.LAMBDA[1] * self.ncr_loss + \
        #             cfg.LAMBDA[2] * self.ed_loss + cfg.LAMBDA[3] * self.et_loss

        self.loss = cfg.LAMBDA[0] * self.bg_loss + cfg.LAMBDA[1] * self.ncr_loss + cfg.LAMBDA[2] * self.ed_loss + cfg.LAMBDA[3] * self.et_loss



    def resnet(self):
        # start down sampling by depth n.
        self.down_conv = [0] * cfg.DEPTH
        self.down_pool = [0] * cfg.DEPTH
        self.up_conv = [0] * cfg.DEPTH
        self.up_pool = [0] * cfg.DEPTH

        with tf.variable_scope('down'):

            inputs = self.X     # iterator 변수 self.features 를 이용해 inputs 생성
            channel_n = cfg.INIT_N_FILTER
            pool_size = cfg.IMG_SIZE

            if cfg.FIRST_DOWNSAMPLING:
                pool_size //= 2
                inputs = utils.select_downsampling('first_downsampling', inputs, [], channel_n, pool_size, cfg.DOWNSAMPLING_TYPE)

            for i in range(cfg.DEPTH):
                pool_size //= 2

                self.down_conv[i] = utils.residual_block_v1(inputs = inputs,
                                                            channel_n = channel_n,
                                                            group_n = cfg.GROUP_N,
                                                            act_fn = cfg.ACTIVATION_FUNC,
                                                            norm_type = cfg.NORMALIZATION_TYPE,
                                                            training = self.training,
                                                            idx = i,
                                                            shortcut = True)
                print('down_conv', self.down_conv[i])

                channel_n *= 2

                self.down_pool[i] = utils.select_downsampling(name = str(i) + '_downsampling',
                                                              down_conv = self.down_conv[i],
                                                              down_pool = self.down_pool[i],
                                                              channel_n = channel_n,
                                                              pool_size = pool_size,
                                                              mode = cfg.DOWNSAMPLING_TYPE)



                inputs = tf.identity(self.down_pool[i])
                print('down_pool', inputs)


            inputs = utils.unet_same_block(inputs = inputs,
                                           channel_n = channel_n,
                                           group_n = cfg.GROUP_N,
                                           act_fn = cfg.ACTIVATION_FUNC,
                                           norm_type = cfg.NORMALIZATION_TYPE,
                                           training = self.training)

        with tf.variable_scope('up'):

            for i in reversed(range(cfg.DEPTH)):
                channel_n //= 2
                pool_size *= 2

                inputs = utils.select_upsampling(name = str(i) + '_upsampling',
                                                 up_conv = inputs,
                                                 up_pool = self.up_pool[i],
                                                 channel_n = channel_n,
                                                 pool_size = pool_size,
                                                 mode = cfg.UPSAMPLING_TYPE)
                print('up_pool', inputs)

                inputs = utils.unet_up_block(inputs = inputs,
                                             downconv_list = self.down_conv,
                                             upconv_list = self.up_conv,
                                             pool_list = self.up_pool,
                                             channel_n = channel_n,
                                             group_n = cfg.GROUP_N,
                                             act_fn = cfg.ACTIVATION_FUNC,
                                             norm_type = cfg.NORMALIZATION_TYPE,
                                             training = self.training,
                                             idx = i)
                print('up_conv', inputs)

            if cfg.FIRST_DOWNSAMPLING:
                channel_n //= 2
                pool_size *= 2
                inputs= utils.select_upsampling('last_upsampling', inputs, [], channel_n, pool_size, cfg.UPSAMPLING_TYPE)

            up_conv_f = utils.conv2D('final_upconv', inputs, 4, [1,1], [1,1], 'SAME')

        return up_conv_f