'''
model construct

by 신형은 주임
'''
import tensorflow as tf
import utils
import config as cfg
# import Unet.aneurysm_unet.merging.utils as utils
# import Unet.aneurysm_unet.merging.config as cfg

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

        self.logit = self.u_net()

        self.pred = tf.nn.softmax(logits=self.logit)
        # 활성화 시킨 probability map을 split 하여 foreground와 background로 분리합니다.
        self.foreground_predicted, self.background_predicted = tf.split(self.pred, [1, 1], 3)

        # 라벨이미지 역시 foreground와 background로 분리합니다
        self.foreground_truth, self.background_truth = tf.split(self.labels, [1, 1], 3)
        self.loss_1 = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.foreground_predicted, target=self.foreground_truth)
        self.loss_2 = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.background_predicted, target=self.background_truth)
        self.loss = cfg.LAMBDA * self.loss_1 + (1 - cfg.LAMBDA) * self.loss_2
        self.results = list(utils.iou_coe(output=self.foreground_predicted, target=self.foreground_truth))


    def u_net(self):
        # start down sampling by depth n.
        self.down_conv = [0] * cfg.DEPTH
        self.down_pool = [0] * cfg.DEPTH
        self.up_conv = [0] * cfg.DEPTH
        self.up_pool = [0] * cfg.DEPTH

        with tf.variable_scope('down'):

            inputs = self.features     # iterator 변수 self.features 를 이용해 inputs 생성
            channel_n = cfg.INIT_N_FILTER
            pool_size = cfg.IMG_SIZE

            if cfg.FIRST_DOWNSAMPLING:
                pool_size //= 2
                inputs = utils.select_downsampling('first_downsampling', inputs, [], channel_n, pool_size, cfg.DOWNSAMPLING_TYPE)

            # 처음 실행하면 모델 구조 나오도록 ?!
            for i in range(cfg.DEPTH):
                pool_size //= 2

                inputs = utils.unet_down_block(inputs = inputs,
                                               conv_list = self.down_conv,
                                               pool_list = self.down_pool,
                                               channel_n = channel_n,
                                               pool_size = pool_size,
                                               group_n = cfg.GROUP_N,
                                               act_fn = cfg.ACTIVATION_FUNC,
                                               norm_type = cfg.NORMALIZATION_TYPE,
                                               down_type = cfg.DOWNSAMPLING_TYPE,
                                               training = self.training,
                                               idx = i)
                channel_n *= 2


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

            up_conv_f = utils.conv2D('final_upconv', inputs, cfg.N_CLASS, [1,1], [1,1], 'SAME')

        return up_conv_f