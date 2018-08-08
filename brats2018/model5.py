import tensorflow as tf
import utils
import config as cfg
# import brats2018.utils as utils


class Model:
    def __init__(self):
        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        self.training = tf.placeholder(tf.bool, name='training')
        self.X = tf.placeholder(tf.float32, [None, None, None, cfg.N_INPUT_CHANNEL], name='X')
        self.Y = tf.placeholder(tf.float32, [None, None, None, cfg.N_CLASS], name='Y')
        self.loss_ratio = tf.placeholder(tf.float32, [len(cfg.LAMBDA)], name='loss_ratio')
        self.logit = self.deeplab()

        self.pred = tf.nn.softmax(logits=self.logit)

        self.bg_pred, self.ncr_pred, self.ed_pred, self.et_pred = tf.split(self.pred, [1,1,1,1], axis=3)
        self.bg_label, self.ncr_label, self.ed_label, self.et_label = tf.split(self.Y, [1,1,1,1], axis=3)

        self.bg_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.bg_pred, target=self.bg_label)
        self.ncr_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.ncr_pred, target=self.ncr_label)
        self.ed_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.ed_pred, target=self.ed_label)
        self.et_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.et_pred, target=self.et_label)
        self.loss = self.loss_ratio[0] * self.bg_loss + self.loss_ratio[1] * self.ncr_loss + \
                    self.loss_ratio[2] * self.ed_loss + self.loss_ratio[3] * self.et_loss



    def deeplab(self):
        self.down_conv = [0] * (cfg.DEPTH + 1)

        with tf.variable_scope('down'):

            inputs = self.X  # iterator 변수 self.features 를 이용해 inputs 생성
            channel_n = cfg.INIT_N_FILTER
            pool_size_h = cfg.PATCH_SIZE // 2
            pool_size_w = cfg.PATCH_SIZE // 2
            print(inputs)
            for i in range(cfg.N_LAYERS[0]):
                inputs = utils.xception_depthwise_separable_convlayer(name='dsconv_0_{}'.format(str(i)),
                                                                      inputs=inputs,
                                                                      channel_n=channel_n,
                                                                      last_stride=1,
                                                                      act_fn=cfg.ACTIVATION_FUNC,
                                                                      training=self.training,
                                                                      shortcut_conv=True,
                                                                      atrous=False)

            inputs = utils.select_downsampling(name='0_downsampling',
                                               down_conv=inputs,
                                               down_pool=[],
                                               channel_n=channel_n,
                                               pool_size_h=pool_size_h,
                                               pool_size_w=pool_size_w,
                                               mode=cfg.DOWNSAMPLING_TYPE)

            self.down_conv[0] = tf.identity(inputs)
            print(inputs)

            for i in range(1, cfg.DEPTH):
                channel_n *= 2
                for j in range(cfg.N_LAYERS[i]-1):
                    inputs = utils.xception_depthwise_separable_convlayer(name='dsconv_{}_{}'.format(str(i), str(j)),
                                                                          inputs=inputs,
                                                                          channel_n=channel_n,
                                                                          last_stride=1,
                                                                          act_fn=cfg.ACTIVATION_FUNC,
                                                                          training=self.training,
                                                                          shortcut_conv=True,
                                                                          atrous=False)
                inputs = utils.xception_depthwise_separable_convlayer(name='dsconv_{}_{}'.format(str(i), str(cfg.N_LAYERS[i] - 1)),
                                                                      inputs=inputs,
                                                                      channel_n=channel_n,
                                                                      last_stride=1,
                                                                      act_fn=cfg.ACTIVATION_FUNC,
                                                                      training=self.training,
                                                                      shortcut_conv=True,
                                                                      atrous=True,
                                                                      atrous_rate=2)
                self.down_conv[i] = tf.identity(inputs)
                print(inputs)

            self.down_conv[-1] = utils.atrous_spatial_pyramid_pooling(name='aspp_layer',
                                                                      inputs=inputs,
                                                                      channel_n=channel_n,
                                                                      atrous_rate_list=[[8,8],[12,12],[16,16]],
                                                                      act_fn=cfg.ACTIVATION_FUNC,
                                                                      training=self.training)
            print(self.down_conv[-1])
        with tf.variable_scope('up'):
            pool_size_h *= 2
            pool_size_w *= 2

            concated_conv = tf.concat([utils.conv2D('concated_conv_{}'.format(idx), dc, cfg.INIT_N_FILTER, [1, 1], [1, 1], padding='SAME')
                                       for idx, dc in enumerate(self.down_conv)], axis=-1)
            print(concated_conv)

            # def depthwise_separable_convlayer(name, inputs, channel_n, width_mul, group_n, act_fn, norm_type, training,
            #                                   idx, rate=None):

            concated_conv = utils.depthwise_separable_convlayer(name='usconv',
                                                                inputs=concated_conv,
                                                                channel_n=cfg.N_CLASS,
                                                                width_mul=cfg.WIDTH_MULTIPLIER,
                                                                group_n=cfg.GROUP_N,
                                                                act_fn=cfg.ACTIVATION_FUNC,
                                                                norm_type=cfg.NORMALIZATION_TYPE,
                                                                training=self.training,
                                                                idx=0)

            print(concated_conv)
            final_conv = utils.select_upsampling(name='upsampling',
                                                 up_conv=concated_conv,
                                                 up_pool=[],
                                                 channel_n=cfg.N_CLASS,
                                                 pool_size_h=pool_size_h,
                                                 pool_size_w=pool_size_w,
                                                 mode=cfg.UPSAMPLING_TYPE)
            print(final_conv)
        return final_conv