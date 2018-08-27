import tensorflow as tf
import utils
import config as cfg

### Deeplab v3 plus based BeVEAM_NET with generalized dice loss and upgraded upsampling ###
# 1 downsampling( (192, 192) -> (96, 96)) + 3 atrous conv + 1 ASPP + 1 upsampling( (96, 96) -> (192, 192) )
# upsampling concated layers : concat (layer1 downsampling output + layer2 downsampling output + ...) -> upsampling
# generalized dice loss : calculate all losses(bg, ncr, ed, et) with multi labels (0,1,2,3)


class Model:
    def __init__(self):
        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        self.training = tf.placeholder(tf.bool, name='training')
        self.X = tf.placeholder(tf.float32, [None, None, None, cfg.N_INPUT_CHANNEL], name='X')
        self.Y = tf.placeholder(tf.float32, [None, None, None, cfg.N_CLASS], name='Y')
        self.loss_ratio = tf.placeholder(tf.float32, [len(cfg.LAMBDA)], name='loss_ratio')

        self.logit = self.BeVEAM_NET()

        self.pred = tf.nn.softmax(logits=self.logit)

        self.bg_pred, self.fg_pred = tf.split(self.pred, [1,1], axis=3)
        self.bg_label, self.fg_label = tf.split(self.Y, [1,1], axis=3)

        self.bg_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.bg_pred, target=self.bg_label)
        self.fg_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.fg_pred, target=self.fg_label)

        if cfg.LOSS_FUNC == 'g_dice':
            self.loss = utils.generalised_dice_loss(self.pred, self.Y)
        else :
            self.loss = self.loss_ratio[0] * self.bg_loss + self.loss_ratio[1] * self.fg_loss


    def BeVEAM_NET(self):
        self.down_conv = [0] * (cfg.DEPTH + 1)

        with tf.variable_scope('down'):
            inputs = self.X
            channel_n = cfg.INIT_N_FILTER
            print(inputs)
            pool_size_h = cfg.PATCH_SIZE // 2
            pool_size_w = cfg.PATCH_SIZE // 2

            for i in range(2):


                for j in range(cfg.N_LAYERS[0]):
                    inputs = utils.xception_depthwise_separable_convlayer_dr(name='dsconv_{}_{}'.format(str(i), str(j)),
                                                                             inputs=inputs,
                                                                             channel_n=channel_n,
                                                                             last_stride=1,
                                                                             drop_rate=self.drop_rate,
                                                                             act_fn=cfg.ACTIVATION_FUNC,
                                                                             training=self.training,
                                                                             shortcut_conv=True,
                                                                             atrous=False)

                inputs = utils.select_downsampling(name='{}_downsampling'.format(i),
                                                   down_conv=inputs,
                                                   down_pool=[],
                                                   channel_n=channel_n,
                                                   pool_size_h=pool_size_h,
                                                   pool_size_w=pool_size_w,
                                                   mode=cfg.DOWNSAMPLING_TYPE)

                self.down_conv[i] = tf.identity(inputs)

                channel_n *= 2
                pool_size_h //= 2
                pool_size_w //= 2
                print(inputs)

            for i in range(2, cfg.DEPTH):
                channel_n *= 2
                for j in range(cfg.N_LAYERS[i]-1):
                    inputs = utils.xception_depthwise_separable_convlayer_dr(name='dsconv_{}_{}'.format(str(i), str(j)),
                                                                             inputs=inputs,
                                                                             channel_n=channel_n,
                                                                             last_stride=1,
                                                                             drop_rate=self.drop_rate,
                                                                             act_fn=cfg.ACTIVATION_FUNC,
                                                                             training=self.training,
                                                                             shortcut_conv=True,
                                                                             atrous=False)
                inputs = utils.xception_depthwise_separable_convlayer_dr(name='dsconv_{}_{}'.format(str(i), str(cfg.N_LAYERS[i] - 1)),
                                                                         inputs=inputs,
                                                                         channel_n=channel_n,
                                                                         last_stride=1,
                                                                         drop_rate=self.drop_rate,
                                                                         act_fn=cfg.ACTIVATION_FUNC,
                                                                         training=self.training,
                                                                         shortcut_conv=True,
                                                                         atrous=True,
                                                                         atrous_rate=2**i)
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
                                       for idx, dc in enumerate(self.down_conv) if idx != 0] , axis=-1)


            concated_conv = utils.depthwise_separable_convlayer_dr(name='usconv0',
                                                                   inputs=concated_conv,
                                                                   channel_n=cfg.INIT_N_FILTER,
                                                                   width_mul=cfg.WIDTH_MULTIPLIER,
                                                                   group_n=cfg.GROUP_N,
                                                                   drop_rate=self.drop_rate,
                                                                   act_fn=cfg.ACTIVATION_FUNC,
                                                                   norm_type=cfg.NORMALIZATION_TYPE,
                                                                   training=self.training,
                                                                   idx=0)
            print(concated_conv)
            concated_conv = utils.select_upsampling(name='upsampling0',
                                                 up_conv=concated_conv,
                                                 up_pool=[],
                                                 channel_n=cfg.INIT_N_FILTER,
                                                 pool_size_h=pool_size_h,
                                                 pool_size_w=pool_size_w,
                                                 mode=cfg.UPSAMPLING_TYPE)
            print(concated_conv)

            concated_conv = tf.concat([concated_conv, self.down_conv[0]], axis=-1)

            concated_conv = utils.depthwise_separable_convlayer_dr(name='usconv1',
                                                                   inputs=concated_conv,
                                                                   channel_n=cfg.N_CLASS,
                                                                   width_mul=cfg.WIDTH_MULTIPLIER,
                                                                   group_n=cfg.GROUP_N,
                                                                   drop_rate=self.drop_rate,
                                                                   act_fn=cfg.ACTIVATION_FUNC,
                                                                   norm_type=cfg.NORMALIZATION_TYPE,
                                                                   training=self.training,
                                                                   idx=0)

            pool_size_h *= 2
            pool_size_w *= 2
            print(concated_conv)
            final_conv = utils.select_upsampling(name='upsampling1',
                                                 up_conv=concated_conv,
                                                 up_pool=[],
                                                 channel_n=cfg.N_CLASS,
                                                 pool_size_h=pool_size_h,
                                                 pool_size_w=pool_size_w,
                                                 mode=cfg.UPSAMPLING_TYPE)

            print(final_conv)
        return final_conv