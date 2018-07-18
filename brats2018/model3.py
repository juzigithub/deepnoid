import tensorflow as tf
import utils
# import brats2018.utils as utils
import config as cfg


class Model:
    def __init__(self):
        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        self.training = tf.placeholder(tf.bool, name='training')
        self.X = tf.placeholder(tf.float32, [None, 192, 160, 4], name='X')
        self.Y = tf.placeholder(tf.float32, [None, 192, 160, 4], name='Y')
        self.logit = self.mobilenet()

        self.pred = tf.nn.softmax(logits=self.logit)

        self.bg_pred, self.ncr_pred, self.ed_pred, self.et_pred = tf.split(self.pred, [1,1,1,1], axis=3)
        self.bg_label, self.ncr_label, self.ed_label, self.et_label = tf.split(self.Y, [1,1,1,1], axis=3)

        self.bg_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.bg_pred, target=self.bg_label)
        self.ncr_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.ncr_pred, target=self.ncr_label)
        self.ed_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.ed_pred, target=self.ed_label)
        self.et_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.et_pred, target=self.et_label)
        self.loss = cfg.LAMBDA[0] * self.bg_loss + cfg.LAMBDA[1] * self.ncr_loss + \
                    cfg.LAMBDA[2] * self.ed_loss + cfg.LAMBDA[3] * self.et_loss



    def mobilenet(self):
        # start down sampling by depth n.
        self.down_conv = [0] * cfg.DEPTH
        self.down_pool = [0] * cfg.DEPTH
        self.up_conv = [0] * cfg.DEPTH
        self.up_pool = [0] * cfg.DEPTH

        with tf.variable_scope('down'):

            inputs = self.X  # iterator 변수 self.features 를 이용해 inputs 생성
            channel_n = cfg.INIT_N_FILTER
            pool_size_h = cfg.IMG_SIZE[0]
            pool_size_w = cfg.IMG_SIZE[1]

            pool_size_h //= 2
            pool_size_w //= 2

            self.down_conv0 = tf.identity(inputs)

            for i in range(cfg.N_LAYERS[0]):
                self.down_conv0 = utils.residual_block_dw_dr(name='downres_0_{}'.format(str(i)),
                                                             inputs=self.down_conv0,
                                                             channel_n=channel_n,
                                                             width_mul=cfg.WIDTH_MULTIPLIER,
                                                             group_n=cfg.GROUP_N,
                                                             drop_rate=cfg.DROPOUT_RATE,
                                                             act_fn=cfg.ACTIVATION_FUNC,
                                                             norm_type=cfg.NORMALIZATION_TYPE,
                                                             training=self.training,
                                                             idx=i)
            print('down_conv0', self.down_conv0)

            self.down_pool0 = utils.select_downsampling2(name='downsampling0',
                                                         down_conv=self.down_conv0,
                                                         channel_n=channel_n,
                                                         pool_size_h=pool_size_h,
                                                         pool_size_w=pool_size_w,
                                                         mode=cfg.DOWNSAMPLING_TYPE)
            print('down_pool0', self.down_pool0)

            channel_n *= 2
            pool_size_h //= 2
            pool_size_w //= 2

            self.down_conv1 = tf.identity(self.down_pool0)

            # def residual_block_dw_dr(name, inputs, channel_n, width_mul, group_n, drop_rate, act_fn, norm_type,
            #                          training, idx, rate=None, shortcut=True):

            for i in range(cfg.N_LAYERS[1]):
                self.down_conv1 = utils.residual_block_dw_dr(name='downres_1_{}'.format(str(i)),
                                                             inputs=self.down_conv1,
                                                             channel_n=channel_n,
                                                             width_mul=cfg.WIDTH_MULTIPLIER,
                                                             group_n=cfg.GROUP_N,
                                                             drop_rate=cfg.DROPOUT_RATE,
                                                             act_fn=cfg.ACTIVATION_FUNC,
                                                             norm_type=cfg.NORMALIZATION_TYPE,
                                                             training=self.training,
                                                             idx=i)
            print('down_conv1', self.down_conv1)

            self.down_pool1 = utils.select_downsampling2(name='downsampling1',
                                                         down_conv=self.down_conv1,
                                                         channel_n=channel_n,
                                                         pool_size_h=pool_size_h,
                                                         pool_size_w=pool_size_w,
                                                         mode=cfg.DOWNSAMPLING_TYPE)
            print('down_pool1', self.down_pool1)

            channel_n *= 2
            pool_size_h //= 2
            pool_size_w //= 2

            self.down_conv2 = tf.identity(self.down_pool1)

            for i in range(cfg.N_LAYERS[2]):
                self.down_conv2 = utils.residual_block_dw_dr(name='downres_2_{}'.format(str(i)),
                                                             inputs=self.down_conv2,
                                                             channel_n=channel_n,
                                                             width_mul=cfg.WIDTH_MULTIPLIER,
                                                             group_n=cfg.GROUP_N,
                                                             drop_rate=cfg.DROPOUT_RATE,
                                                             act_fn=cfg.ACTIVATION_FUNC,
                                                             norm_type=cfg.NORMALIZATION_TYPE,
                                                             training=self.training,
                                                             idx=i)
            print('down_conv2', self.down_conv2)

            self.down_pool2 = utils.select_downsampling2(name='downsampling2',
                                                         down_conv=self.down_conv2,
                                                         channel_n=channel_n,
                                                         pool_size_h=pool_size_h,
                                                         pool_size_w=pool_size_w,
                                                         mode=cfg.DOWNSAMPLING_TYPE)
            print('down_pool2', self.down_pool2)

            channel_n *= 2
            pool_size_h //= 2
            pool_size_w //= 2

            self.down_conv3 = tf.identity(self.down_pool2)

            for i in range(cfg.N_LAYERS[3]):
                self.down_conv3 = utils.residual_block_dw_dr(name='downres_3_{}'.format(str(i)),
                                                             inputs=self.down_conv3,
                                                             channel_n=channel_n,
                                                             width_mul=cfg.WIDTH_MULTIPLIER,
                                                             group_n=cfg.GROUP_N,
                                                             drop_rate=cfg.DROPOUT_RATE,
                                                             act_fn=cfg.ACTIVATION_FUNC,
                                                             norm_type=cfg.NORMALIZATION_TYPE,
                                                             training=self.training,
                                                             idx=i)
            print('down_conv3', self.down_conv3)

            self.down_pool3 = utils.select_downsampling2(name='downsampling3',
                                                         down_conv=self.down_conv3,
                                                         channel_n=channel_n,
                                                         pool_size_h=pool_size_h,
                                                         pool_size_w=pool_size_w,
                                                         mode=cfg.DOWNSAMPLING_TYPE)
            print('down_pool3', self.down_pool3)

            self.same_conv = tf.identity(self.down_pool3)

            for i in range(1):
                self.same_conv = utils.residual_block_dw_dr(name='sameres_{}'.format(str(i)),
                                                            inputs=self.same_conv,
                                                            channel_n=channel_n,
                                                            width_mul=cfg.WIDTH_MULTIPLIER,
                                                            group_n=cfg.GROUP_N,
                                                            drop_rate=cfg.DROPOUT_RATE,
                                                            act_fn=cfg.ACTIVATION_FUNC,
                                                            norm_type=cfg.NORMALIZATION_TYPE,
                                                            training=self.training,
                                                            idx=i)
            print('same_conv', self.same_conv)
            pool_size_h *= 2
            pool_size_w *= 2

            self.up_pool3 = utils.select_upsampling2(name='upsampling3',
                                                     up_conv=utils.concat('uppool_3', [self.down_pool3, self.same_conv], axis=-1),
                                                     channel_n=channel_n,
                                                     pool_size_h=pool_size_h,
                                                     pool_size_w=pool_size_w,
                                                     mode=cfg.UPSAMPLING_TYPE)

            print('up_pool3', self.up_pool3)

            del self.down_pool3, self.same_conv

            self.up_conv3 = utils.concat('up_conv3', [self.down_conv3, self.up_pool3], axis=-1)

            del self.down_conv3, self.up_pool3 ###########################

            channel_n //= 2
            for i in range(1):
                self.up_conv3 = utils.residual_block_dw_dr(name='upres_3_{}'.format(str(i)),
                                                             inputs=self.up_conv3,
                                                             channel_n=channel_n,
                                                           width_mul=cfg.WIDTH_MULTIPLIER,
                                                           group_n=cfg.GROUP_N,
                                                             drop_rate=cfg.DROPOUT_RATE,
                                                             act_fn=cfg.ACTIVATION_FUNC,
                                                             norm_type=cfg.NORMALIZATION_TYPE,
                                                             training=self.training,
                                                             idx=i)
            print('up_conv3', self.up_conv3)
            pool_size_h *= 2
            pool_size_w *= 2

            self.up_pool2 = utils.select_upsampling2(name='upsampling2',
                                                     up_conv=utils.concat('uppool_2', [self.down_pool2, self.up_conv3], axis=-1),
                                                     channel_n=channel_n,
                                                     pool_size_h=pool_size_h,
                                                     pool_size_w=pool_size_w,
                                                     mode=cfg.UPSAMPLING_TYPE)
            print('up_pool2', self.up_pool2)

            del self.down_pool2, self.up_conv3

            self.up_conv2 = utils.concat('up_conv2', [self.down_conv2, self.up_pool2], axis=-1)

            del self.down_conv2, self.up_pool2 ###########################

            channel_n //= 2
            for i in range(1):
                self.up_conv2 = utils.residual_block_dw_dr(name='upres_2_{}'.format(str(i)),
                                                             inputs=self.up_conv2,
                                                             channel_n=channel_n,
                                                           width_mul=cfg.WIDTH_MULTIPLIER,
                                                           group_n=cfg.GROUP_N,
                                                             drop_rate=cfg.DROPOUT_RATE,
                                                             act_fn=cfg.ACTIVATION_FUNC,
                                                             norm_type=cfg.NORMALIZATION_TYPE,
                                                             training=self.training,
                                                             idx=i)
            print('up_conv2', self.up_conv2)

            pool_size_h *= 2
            pool_size_w *= 2

            self.up_pool1 = utils.select_upsampling2(name='upsampling1',
                                                     up_conv=utils.concat('uppool_1', [self.down_pool1, self.up_conv2], axis=-1),
                                                     channel_n=channel_n,
                                                     pool_size_h=pool_size_h,
                                                     pool_size_w=pool_size_w,
                                                     mode=cfg.UPSAMPLING_TYPE)
            print('up_pool1', self.up_pool1)

            del self.down_pool1, self.up_conv2

            self.up_conv1 = utils.concat('up_conv1', [self.down_conv1, self.up_pool1], axis=-1)

            del self.down_conv1, self.up_pool1 ###########################

            channel_n //= 2
            for i in range(1):
                self.up_conv1 = utils.residual_block_dw_dr(name='upres_1_{}'.format(str(i)),
                                                             inputs=self.up_conv1,
                                                             channel_n=channel_n,
                                                           width_mul=cfg.WIDTH_MULTIPLIER,
                                                           group_n=cfg.GROUP_N,
                                                             drop_rate=cfg.DROPOUT_RATE,
                                                             act_fn=cfg.ACTIVATION_FUNC,
                                                             norm_type=cfg.NORMALIZATION_TYPE,
                                                             training=self.training,
                                                             idx=i)
            print('up_conv1', self.up_conv1)
            pool_size_h *= 2
            pool_size_w *= 2

            self.up_pool0 = utils.select_upsampling2(name='upsampling0',
                                                     up_conv=utils.concat('uppool_0', [self.down_pool0, self.up_conv1], axis=-1),
                                                     channel_n=channel_n,
                                                     pool_size_h=pool_size_h,
                                                     pool_size_w=pool_size_w,
                                                     mode=cfg.UPSAMPLING_TYPE)
            print('up_pool0', self.up_pool0)

            del self.down_pool0, self.up_conv1

            self.up_conv0 = utils.concat('up_conv0', [self.down_conv0, self.up_pool0], axis=-1)

            del self.down_conv0, self.up_pool0 ###########################


            for i in range(1):
                self.up_conv0 = utils.residual_block_dw_dr(name='upres_0_{}'.format(str(i)),
                                                             inputs=self.up_conv0,
                                                             channel_n=channel_n,
                                                           width_mul=cfg.WIDTH_MULTIPLIER,
                                                           group_n=cfg.GROUP_N,
                                                             drop_rate=cfg.DROPOUT_RATE,
                                                             act_fn=cfg.ACTIVATION_FUNC,
                                                             norm_type=cfg.NORMALIZATION_TYPE,
                                                             training=self.training,
                                                             idx=i)
            print('up_conv0', self.up_conv0)
            up_conv_f = utils.conv2D('final_upconv', self.up_conv0, cfg.N_CLASS, [1, 1], [1, 1], 'SAME')

            del self.up_conv0

            print('final_conv', up_conv_f)

        return up_conv_f