'''
model construct

by 신형은 주임
'''
import tensorflow as tf
# import utils
# import config as cfg
import Unet.aneurysm_unet.merging.utils as utils
import Unet.aneurysm_unet.merging.config as cfg


class Model:
    def __init__(self):
        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        self.training = tf.placeholder(tf.bool, name='training')
        self.X = tf.placeholder(tf.float32, [None, cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.RGB_CHANNEL], name='X')
        self.Y = tf.placeholder(tf.float32, [None, cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.N_CLASS], name='Y')
        self.X_ADD = tf.placeholder(tf.int32, [None, 3], name='X_ADD')

        # iterator 설정
        self.dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y, self.X_ADD)).shuffle(buffer_size=cfg.BUFFER_SIZE)
        self.dataset = self.dataset.batch(cfg.BATCH_SIZE).repeat()
        # self.features, self.labels, self.address 를 iterator 변수로 설정 -> iterator 변수 호출 시 다음 데이터를 불러옵니다.
        self.iter = self.dataset.make_initializable_iterator()
        self.features, self.labels, self.address = self.iter.get_next()  #### self.X, self.Y 들어갈 자리에 self.features, self.labels 입력

        self.logit = self.deeplab()

        self.pred = tf.nn.softmax(logits=self.logit)
        # 활성화 시킨 probability map을 split 하여 foreground와 background로 분리합니다.
        self.foreground_predicted, self.background_predicted = tf.split(self.pred, [1, 1], 3)

        # 라벨이미지 역시 foreground와 background로 분리합니다
        self.foreground_truth, self.background_truth = tf.split(self.labels, [1, 1], 3)
        self.loss_1 = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.foreground_predicted, target=self.foreground_truth)
        self.loss_2 = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.background_predicted, target=self.background_truth)
        self.loss = cfg.LAMBDA * self.loss_1 + (1 - cfg.LAMBDA) * self.loss_2
        self.results = list(utils.iou_coe(output=self.foreground_predicted, target=self.foreground_truth))


    def deeplab(self):
        # start down sampling by depth n.
        self.down_conv = [0] * cfg.DEPTH
        self.down_pool = [0] * cfg.DEPTH
        self.up_conv = [0] * cfg.DEPTH
        self.up_pool = [0] * cfg.DEPTH

        with tf.variable_scope('down'):

            inputs = self.features     # iterator 변수 self.features 를 이용해 inputs 생성
            channel_n = cfg.INIT_N_FILTER

            # input conv layer
            inputs = utils.conv2D('first_downconv', inputs, channel_n, [3, 3], [2, 2], padding='SAME')
            inputs = utils.Normalization(inputs, 'batch', self.training, 'first_downconv_norm')
            inputs = utils.activation('first_downconv_act', inputs, cfg.ACTIVATION_FUNC)

            # entry xception layer
            layer = utils.xception_depthwise_separable_convlayer(name = 'entry1',
                                                                 inputs = inputs,
                                                                 channel_n = channel_n * 2,
                                                                 last_stride = 2,
                                                                 act_fn = cfg.ACTIVATION_FUNC,
                                                                 training = self.training,
                                                                 shortcut_conv = True,
                                                                 atrous = False)

            low_level_features = tf.identity(layer)

            layer = utils.xception_depthwise_separable_convlayer(name = 'entry2',
                                                                 inputs = layer,
                                                                 channel_n = channel_n * 4,
                                                                 last_stride = 2,
                                                                 act_fn = cfg.ACTIVATION_FUNC,
                                                                 training = self.training,
                                                                 shortcut_conv = True,
                                                                 atrous = False)

            layer = utils.xception_depthwise_separable_convlayer(name = 'entry3',
                                                                 inputs = layer,
                                                                 channel_n = channel_n * 16,
                                                                 last_stride = 2,
                                                                 act_fn = cfg.ACTIVATION_FUNC,
                                                                 training = self.training,
                                                                 shortcut_conv = True,
                                                                 atrous = False)

            for i in range(cfg.MIDDLE_REPEAT):
                layer = utils.xception_depthwise_separable_convlayer(name = 'middle' + str(i),
                                                                     inputs = layer,
                                                                     channel_n = channel_n * 16,
                                                                     last_stride = 1,
                                                                     act_fn = cfg.ACTIVATION_FUNC,
                                                                     training = self.training,
                                                                     shortcut_conv = False,
                                                                     atrous = True)

            # exit conv layer
            layer = utils.xception_depthwise_separable_convlayer(name = 'exit',
                                                                 inputs = layer,
                                                                 channel_n = channel_n * 16,
                                                                 last_stride = 1,
                                                                 act_fn = cfg.ACTIVATION_FUNC,
                                                                 training = self.training,
                                                                 shortcut_conv = True,
                                                                 atrous = False)


            layer = utils.conv2D('exit_downconv', layer, channel_n * 16, [3, 3], [1, 1], padding='SAME')
            layer = utils.Normalization(layer, 'batch', self.training, 'exit_downconv_norm')
            layer = utils.activation('exit_downconv_act', layer, cfg.ACTIVATION_FUNC)

            # aspp layer
            layer = utils.atrous_spatial_pyramid_pooling(name = 'aspp',
                                                         inputs = layer,
                                                         channel_n = channel_n * 2,
                                                         output_stride = 16,
                                                         act_fn = cfg.ACTIVATION_FUNC,
                                                         training = self.training)

        with tf.variable_scope('up'):

            # aspp -> x4 upsample
            layer = utils.select_upsampling(name = 'x4_upsample1',
                                            up_conv = layer,
                                            up_pool = [],
                                            channel_n = channel_n * 2,
                                            pool_size = layer.get_shape().as_list()[-1] * 4,
                                            mode = cfg.UPSAMPLING_TYPE)
            layer = utils.Normalization(layer, 'batch', self.training, 'x4_upsample_norm1')
            layer = utils.activation('x4_upsample_act1', layer, cfg.ACTIVATION_FUNC)

            # low-level-features -> 1x1 conv
            low_level_features = utils.conv2D('low_level_conv', low_level_features, channel_n * 2, [1, 1], [1, 1], padding='SAME')
            low_level_features = utils.Normalization(low_level_features, 'batch', self.training, 'low_level_norm')
            low_level_features = utils.activation('low_level_act', low_level_features, cfg.ACTIVATION_FUNC)

            # concat aspp & low-level-features
            layer = tf.concat([layer, low_level_features], axis=3)

            # 3x3 conv followed by x4 upsample
            layer = utils.conv2D('last_conv_layer', layer, cfg.N_CLASS, [3, 3], [1, 1], padding='SAME')
            layer = utils.Normalization(layer, 'batch', self.training, 'last_conv_norm')
            layer = utils.activation('last_conv_act', layer, cfg.ACTIVATION_FUNC)

            layer = utils.select_upsampling(name = 'x4_upsample2',
                                            up_conv = layer,
                                            up_pool = [],
                                            channel_n = cfg.N_CLASS,
                                            pool_size = cfg.IMG_SIZE,
                                            mode = cfg.UPSAMPLING_TYPE)
            layer = utils.Normalization(layer, 'batch', self.training, 'x4_upsample_norm2')
            layer = utils.activation('x4_upsample_act2', layer, cfg.ACTIVATION_FUNC)

        return layer

        # if cfg.FIRST_DOWNSAMPLING:
                # pool_size //= 2
                # inputs = utils.select_downsampling('first_downsampling', inputs, [], channel_n, pool_size, cfg.DOWNSAMPLING_TYPE)


        #
        #
        #     for i in range(cfg.DEPTH):
        #         pool_size //= 2
        #         self.down_conv[i] = utils.depthwise_separable_convlayer(name = 'dsconv' + str(i),
        #                                                                 inputs = inputs,
        #                                                                 channel_n = channel_n,
        #                                                                 width_mul = cfg.WIDTH_MULTIPLIER,
        #                                                                 group_n = cfg.GROUP_N,
        #                                                                 act_fn = cfg.ACTIVATION_FUNC,
        #                                                                 norm_type = cfg.NORMALIZATION_TYPE,
        #                                                                 training = self.training,
        #                                                                 idx = i)
        #         print('down_conv', self.down_conv[i])
        #         channel_n *= 2
        #         self.down_pool[i] = utils.select_downsampling(name = str(i) + '_downsampling',
        #                                                       down_conv = self.down_conv[i],
        #                                                       down_pool = self.down_pool[i],
        #                                                       channel_n = channel_n,
        #                                                       pool_size = pool_size,
        #                                                       mode = cfg.DOWNSAMPLING_TYPE)
        #
        #         inputs = tf.identity(self.down_pool[i])
        #         print('down_pool', inputs)
        #
        #     inputs = utils.unet_same_block(inputs = inputs,
        #                                    channel_n = channel_n,
        #                                    group_n = cfg.GROUP_N,
        #                                    act_fn = cfg.ACTIVATION_FUNC,
        #                                    norm_type = cfg.NORMALIZATION_TYPE,
        #                                    training = self.training)
        #
        # with tf.variable_scope('up'):
        #
        #     for i in reversed(range(cfg.DEPTH)):
        #         channel_n //= 2
        #         pool_size *= 2
        #         inputs = utils.select_upsampling(name = str(i) + '_upsampling',
        #                                          up_conv = inputs,
        #                                          up_pool = self.up_pool[i],
        #                                          channel_n = channel_n,
        #                                          pool_size = pool_size,
        #                                          mode = cfg.UPSAMPLING_TYPE)
        #         print('up_pool', inputs)
        #
        #         inputs = utils.unet_up_block(inputs = inputs,
        #                                      downconv_list = self.down_conv,
        #                                      upconv_list = self.up_conv,
        #                                      pool_list = self.up_pool,
        #                                      channel_n = channel_n,
        #                                      group_n = cfg.GROUP_N,
        #                                      act_fn = cfg.ACTIVATION_FUNC,
        #                                      norm_type = cfg.NORMALIZATION_TYPE,
        #                                      training = self.training,
        #                                      idx = i)
        #         print('up_conv', inputs)
        #
        #     if cfg.FIRST_DOWNSAMPLING:
        #         channel_n //= 2
        #         pool_size *= 2
        #         inputs= utils.select_upsampling('last_upsampling', inputs, [], channel_n, pool_size, cfg.UPSAMPLING_TYPE)
        #
        #     up_conv_f = utils.conv2D('final_upconv', inputs, cfg.N_CLASS, [1,1], [1,1], 'SAME')
        #
        # return up_conv_f