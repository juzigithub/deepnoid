import tensorflow as tf
import utils
import config as cfg

### UNet based Multiscale Segnet with dynamic binary loss ###
# High_path and low_path both predict <bg, ncr, ed, et>

## example ##
# high_path inputs / outputs : (n, 192, 192, 4 <flair, t1, t1ce, t2>) -> (n, 192, 192, 4 <bg, ncr, ed, et>)
# low_path inputs / outputs : (n*4, 96, 96 , 4 <flair, t1, t1ce, t2>) -> (n, 192, 192, 4 <bg, ncr, ed, et>)


class Model:
    def __init__(self):
        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        self.training = tf.placeholder(tf.bool, name='training')
        self.X = tf.placeholder(tf.float32, [None, None, None, cfg.N_INPUT_CHANNEL], name='X')
        self.Y = tf.placeholder(tf.float32, [None, None, None, cfg.N_CLASS], name='Y')
        self.loss_ratio = tf.placeholder(tf.float32, [len(cfg.LAMBDA)], name='loss_ratio')
        self.logit_high, self.logit_low = self.multiscale_segnet()

        self.pred_high = tf.nn.softmax(logits=self.logit_high)
        self.pred_low = tf.nn.softmax(logits=self.logit_low)

        self.bg_pred_high, self.ncr_pred_high, self.ed_pred_high, self.et_pred_high = tf.split(self.pred_high, [1,1,1,1], axis=3)
        self.bg_pred_low, self.ncr_pred_low, self.ed_pred_low, self.et_pred_low = tf.split(self.pred_low, [1,1,1,1], axis=3)
        self.bg_label, self.ncr_label, self.ed_label, self.et_label = tf.split(self.Y, [1,1,1,1], axis=3)

        self.bg_pred = tf.add(tf.scalar_mul(0.8, self.bg_pred_high), tf.scalar_mul(0.2, self.bg_pred_low))
        self.ncr_pred = tf.add(tf.scalar_mul(0.2, self.ncr_pred_high), tf.scalar_mul(0.8, self.ncr_pred_low))
        self.ed_pred = tf.add(tf.scalar_mul(0.2, self.ed_pred_high), tf.scalar_mul(0.8, self.ed_pred_low))
        self.et_pred = tf.add(tf.scalar_mul(0.8, self.et_pred_high), tf.scalar_mul(0.2, self.et_pred_low))

        self.pred = tf.nn.softmax(tf.concat([self.bg_pred, self.ncr_pred, self.ed_pred, self.et_pred], -1))

        self.bg_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.bg_pred, target=self.bg_label)
        self.ncr_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.ncr_pred, target=self.ncr_label)
        self.ed_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.ed_pred, target=self.ed_label)
        self.et_loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.et_pred, target=self.et_label)
        self.loss = self.loss_ratio[0] * self.bg_loss + self.loss_ratio[1] * self.ncr_loss + \
                    self.loss_ratio[2] * self.ed_loss + self.loss_ratio[3] * self.et_loss



    def multiscale_segnet(self):
        # start down sampling by depth n.
        self.down_conv_high = [0] * cfg.DEPTH
        self.down_pool_high = [0] * cfg.DEPTH
        self.up_conv_high = [0] * cfg.DEPTH
        self.up_pool_high = [0] * cfg.DEPTH

        self.down_conv_low = [0] * cfg.DEPTH
        self.down_pool_low = [0] * cfg.DEPTH
        self.up_conv_low = [0] * cfg.DEPTH
        self.up_pool_low = [0] * cfg.DEPTH


        with tf.variable_scope('high_path'):
            inputs_high = self.X
            channel_n = cfg.INIT_N_FILTER
            pool_size_h = cfg.PATCH_SIZE
            pool_size_w = cfg.PATCH_SIZE

            for i in range(cfg.DEPTH_HIGH):
                pool_size_h //= 2
                pool_size_w //= 2
                for j in range(cfg.N_LAYERS_HIGH[i]):
                    inputs_high = utils.residual_block_v1_dr(name='high_resconv_{}_{}_'.format(str(i),str(j)),
                                                                               inputs=inputs_high,
                                                                               channel_n=channel_n,
                                                                               group_n=cfg.GROUP_N,
                                                                               drop_rate=self.drop_rate,
                                                                               act_fn=cfg.ACTIVATION_FUNC,
                                                                               norm_type=cfg.NORMALIZATION_TYPE,
                                                                               training=self.training,
                                                                               idx=i)
                self.down_conv_high[i] = tf.identity(inputs_high)
                print('down_conv_high', self.down_conv_high[i])
                channel_n *= 2
                self.down_pool_high[i] = utils.select_downsampling(name=str(i) + '_high_downsampling',
                                                              down_conv=self.down_conv_high[i],
                                                              down_pool=self.down_pool_high[i],
                                                              channel_n=channel_n,
                                                              pool_size_h=pool_size_h,
                                                              pool_size_w=pool_size_w,
                                                              mode=cfg.DOWNSAMPLING_TYPE)

                inputs_high = tf.identity(self.down_pool_high[i])
                print('down_pool_high', inputs_high)

            inputs_high = utils.unet_same_block(name='high_',
                                                inputs=inputs_high,
                                                channel_n=channel_n,
                                                group_n=cfg.GROUP_N,
                                                act_fn=cfg.ACTIVATION_FUNC,
                                                norm_type=cfg.NORMALIZATION_TYPE,
                                                training=self.training)

            for i in reversed(range(cfg.DEPTH_HIGH)):
                channel_n //= 2
                pool_size_h *= 2
                pool_size_w *= 2
                inputs_high = utils.select_upsampling(name=str(i) + 'high_upsampling',
                                                 up_conv=inputs_high,
                                                 up_pool=self.up_pool_high[i],
                                                 channel_n=channel_n,
                                                 pool_size_h=pool_size_h,
                                                 pool_size_w=pool_size_w,
                                                 mode=cfg.UPSAMPLING_TYPE)
                print('up_pool_high', inputs_high)

                inputs_high = utils.unet_up_block(name='high_',
                                                  inputs=inputs_high,
                                                  downconv_list=self.down_conv_high,
                                                  upconv_list=self.up_conv_high,
                                                  pool_list=self.up_pool_high,
                                                  channel_n=channel_n,
                                                  group_n=cfg.GROUP_N,
                                                  act_fn=cfg.ACTIVATION_FUNC,
                                                  norm_type=cfg.NORMALIZATION_TYPE,
                                                  training=self.training,
                                                  idx=i)
                print('up_conv_high', inputs_high)

            up_conv_f_high = utils.conv2D('final_upconv_high', inputs_high, cfg.N_CLASS, [1, 1], [1, 1], 'SAME')

        with tf.variable_scope('low_path'):
            n, h, c = cfg.BATCH_SIZE, cfg.PATCH_SIZE, cfg.N_INPUT_CHANNEL
            channel_n = cfg.INIT_N_FILTER
            pool_size_h = cfg.PATCH_SIZE // 2
            pool_size_w = cfg.PATCH_SIZE // 2
            p = cfg.PATCH_SIZE // 2

            # Image to Patches Conversion
            pad = [[0, 0], [0, 0]]
            inputs_low = tf.space_to_batch_nd(self.X, [p, p], pad)
            inputs_low = tf.split(inputs_low, p * p, 0)
            inputs_low = tf.stack(inputs_low, 3)
            inputs_low = tf.reshape(inputs_low, [-1, p, p, c])

            for i in range(cfg.DEPTH_LOW):
                pool_size_h //= 2
                pool_size_w //= 2
                for j in range(cfg.N_LAYERS_LOW[i]):
                    inputs_low = utils.residual_block_v1_dr(
                        name='low_resconv_{}_{}_'.format(str(i), str(j)),
                        inputs=inputs_low,
                        channel_n=channel_n,
                        group_n=cfg.GROUP_N,
                        drop_rate=self.drop_rate,
                        act_fn=cfg.ACTIVATION_FUNC,
                        norm_type=cfg.NORMALIZATION_TYPE,
                        training=self.training,
                        idx=i)

                self.down_conv_low[i] = tf.identity(inputs_low)
                print('down_conv_low', self.down_conv_low[i])
                channel_n *= 2
                self.down_pool_low[i] = utils.select_downsampling(name=str(i) + '_low_downsampling',
                                                                   down_conv=self.down_conv_low[i],
                                                                   down_pool=self.down_pool_low[i],
                                                                   channel_n=channel_n,
                                                                   pool_size_h=pool_size_h,
                                                                   pool_size_w=pool_size_w,
                                                                   mode=cfg.DOWNSAMPLING_TYPE)

                inputs_low = tf.identity(self.down_pool_low[i])
                print('down_pool_low', inputs_low)

            inputs_low = utils.unet_same_block(name='low_',
                                                inputs=inputs_low,
                                                channel_n=channel_n,
                                                group_n=cfg.GROUP_N,
                                                act_fn=cfg.ACTIVATION_FUNC,
                                                norm_type=cfg.NORMALIZATION_TYPE,
                                                training=self.training)

            for i in reversed(range(cfg.DEPTH_LOW)):
                channel_n //= 2
                pool_size_h *= 2
                pool_size_w *= 2
                inputs_low = utils.select_upsampling(name=str(i) + 'low_upsampling',
                                                      up_conv=inputs_low,
                                                      up_pool=self.up_pool_low[i],
                                                      channel_n=channel_n,
                                                      pool_size_h=pool_size_h,
                                                      pool_size_w=pool_size_w,
                                                      mode=cfg.UPSAMPLING_TYPE)
                print('up_pool_low', inputs_low)

                inputs_low = utils.unet_up_block(name='low_',
                                                  inputs=inputs_low,
                                                  downconv_list=self.down_conv_low,
                                                  upconv_list=self.up_conv_low,
                                                  pool_list=self.up_pool_low,
                                                  channel_n=channel_n,
                                                  group_n=cfg.GROUP_N,
                                                  act_fn=cfg.ACTIVATION_FUNC,
                                                  norm_type=cfg.NORMALIZATION_TYPE,
                                                  training=self.training,
                                                  idx=i)
                print('up_conv_low', inputs_low)

            up_conv_f_low = utils.conv2D('final_upconv_low', inputs_low, cfg.N_CLASS, [1, 1], [1, 1], 'SAME')
            # Using patches here to reconstruct
            up_conv_f_low = tf.reshape(up_conv_f_low, [-1, h // p, h // p, p * p, c])
            up_conv_f_low = tf.split(up_conv_f_low, p * p, 3)
            up_conv_f_low = tf.stack(up_conv_f_low, axis=0)
            up_conv_f_low = tf.reshape(up_conv_f_low, [-1, h // p, h // p, c])
            up_conv_f_low = tf.batch_to_space_nd(up_conv_f_low, [p, p], pad)

        return up_conv_f_high, up_conv_f_low