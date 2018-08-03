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
        self.logit = self.dual_framelets_resnet()

        self.pred = tf.nn.softmax(logits=self.logit)

        self.bg_pred, *_ = tf.split(self.pred, [1,1,1,1], axis=3)
        self.bg_label, *_= tf.split(self.Y, [1,1,1,1], axis=3)

        self.bg_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.bg_pred, target=self.bg_label)
        self.loss = self.bg_loss



    def dual_framelets_resnet(self):
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

            self.inputs = utils.select_upsampling2(name='upsampling3',
                                                     up_conv=utils.concat('uppool_3', [self.down_pool3, self.same_conv], axis=-1),
                                                     channel_n=channel_n,
                                                     pool_size_h=pool_size_h,
                                                     pool_size_w=pool_size_w,
                                                     mode=cfg.UPSAMPLING_TYPE)

            print('up_pool3', self.inputs)

            del self.down_pool3, self.same_conv

            self.inputs = utils.concat('up_conv3', [self.down_conv3, self.inputs], axis=-1)

            del self.down_conv3

            channel_n //= 2
            for i in range(1):
                self.inputs = utils.residual_block_dw_dr(name='upres_3_{}'.format(str(i)),
                                                             inputs=self.inputs,
                                                             channel_n=channel_n,
                                                           width_mul=cfg.WIDTH_MULTIPLIER,
                                                           group_n=cfg.GROUP_N,
                                                             drop_rate=cfg.DROPOUT_RATE,
                                                             act_fn=cfg.ACTIVATION_FUNC,
                                                             norm_type=cfg.NORMALIZATION_TYPE,
                                                             training=self.training,
                                                             idx=i)
            print('up_conv3', self.inputs)
            pool_size_h *= 2
            pool_size_w *= 2

            self.inputs = utils.select_upsampling2(name='upsampling2',
                                                     up_conv=utils.concat('uppool_2', [self.down_pool2, self.inputs], axis=-1),
                                                     channel_n=channel_n,
                                                     pool_size_h=pool_size_h,
                                                     pool_size_w=pool_size_w,
                                                     mode=cfg.UPSAMPLING_TYPE)
            print('up_pool2', self.inputs)

            del self.down_pool2

            self.inputs = utils.concat('up_conv2', [self.down_conv2, self.inputs], axis=-1)

            del self.down_conv2

            channel_n //= 2
            for i in range(1):
                self.inputs = utils.residual_block_dw_dr(name='upres_2_{}'.format(str(i)),
                                                             inputs=self.inputs,
                                                             channel_n=channel_n,
                                                           width_mul=cfg.WIDTH_MULTIPLIER,
                                                           group_n=cfg.GROUP_N,
                                                             drop_rate=cfg.DROPOUT_RATE,
                                                             act_fn=cfg.ACTIVATION_FUNC,
                                                             norm_type=cfg.NORMALIZATION_TYPE,
                                                             training=self.training,
                                                             idx=i)
            print('up_conv2', self.inputs)

            pool_size_h *= 2
            pool_size_w *= 2

            self.inputs = utils.select_upsampling2(name='upsampling1',
                                                     up_conv=utils.concat('uppool_1', [self.down_pool1, self.inputs], axis=-1),
                                                     channel_n=channel_n,
                                                     pool_size_h=pool_size_h,
                                                     pool_size_w=pool_size_w,
                                                     mode=cfg.UPSAMPLING_TYPE)
            print('up_pool1', self.inputs)

            del self.down_pool1

            self.inputs = utils.concat('up_conv1', [self.down_conv1, self.inputs], axis=-1)

            del self.down_conv1

            channel_n //= 2
            for i in range(1):
                self.inputs = utils.residual_block_dw_dr(name='upres_1_{}'.format(str(i)),
                                                             inputs=self.inputs,
                                                             channel_n=channel_n,
                                                           width_mul=cfg.WIDTH_MULTIPLIER,
                                                           group_n=cfg.GROUP_N,
                                                             drop_rate=cfg.DROPOUT_RATE,
                                                             act_fn=cfg.ACTIVATION_FUNC,
                                                             norm_type=cfg.NORMALIZATION_TYPE,
                                                             training=self.training,
                                                             idx=i)
            print('up_conv1', self.inputs)
            pool_size_h *= 2
            pool_size_w *= 2

            self.inputs= utils.select_upsampling2(name='upsampling0',
                                                     up_conv=utils.concat('uppool_0', [self.down_pool0, self.inputs], axis=-1),
                                                     channel_n=channel_n,
                                                     pool_size_h=pool_size_h,
                                                     pool_size_w=pool_size_w,
                                                     mode=cfg.UPSAMPLING_TYPE)
            print('up_pool0', self.inputs)

            del self.down_pool0

            self.inputs = utils.concat('up_conv0', [self.down_conv0, self.inputs], axis=-1)

            del self.down_conv0


            for i in range(1):
                self.inputs = utils.residual_block_dw_dr(name='upres_0_{}'.format(str(i)),
                                                             inputs=self.inputs,
                                                             channel_n=channel_n,
                                                           width_mul=cfg.WIDTH_MULTIPLIER,
                                                           group_n=cfg.GROUP_N,
                                                             drop_rate=cfg.DROPOUT_RATE,
                                                             act_fn=cfg.ACTIVATION_FUNC,
                                                             norm_type=cfg.NORMALIZATION_TYPE,
                                                             training=self.training,
                                                             idx=i)
            print('up_conv0', self.inputs)
            self.inputs = utils.conv2D('final_upconv', self.inputs, cfg.N_CLASS, [1, 1], [1, 1], 'SAME')

            print('final_conv', self.inputs)

        return self.inputs