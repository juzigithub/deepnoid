'''
model construct

by 신형은 주임
'''
import tensorflow as tf
import utils
import config as cfg


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
##################################################
        self.loss_1 = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.foreground_predicted, target=self.foreground_truth)
        self.loss_2 = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.background_predicted, target=self.background_truth)
        self.loss = self.loss_1 + self.loss_2
##################################################
        self.results = list(utils.iou_coe(output=self.foreground_predicted, target=self.foreground_truth))


    def u_net(self):
        # start down sampling by depth n.

###############################
        self.down_conv = [0] * cfg.DEPTH
        self.down_pool = [0] * cfg.DEPTH
        self.up_conv = [0] * cfg.DEPTH
        self.up_pool = [0] * cfg.DEPTH
###############################

        with tf.variable_scope('down'):

            inputs = self.features     # iterator 변수 self.features 를 이용해 inputs 생성
            channel_n = cfg.INIT_N_FILTER
            pool_size = cfg.IMG_SIZE

            # 처음 실행하면 모델 구조 나오도록 ?!
            for i in range(cfg.DEPTH):
                inputs = utils.unet_down_block(inputs, self.down_conv, self.down_pool, channel_n, pool_size, cfg.GROUP_N, self.training, i)
                channel_n *= 2
                pool_size //= 2

            inputs = utils.unet_same_block(inputs, channel_n, cfg.GROUP_N, self.training)

        with tf.variable_scope('up'):

            for i in reversed(range(cfg.DEPTH)):
                channel_n //= 2
                pool_size *= 2
                inputs = utils.unet_up_block(inputs, self.down_conv, self.up_conv, self.up_pool, channel_n, pool_size, cfg.GROUP_N, self.training, i)

            up_conv_f = utils.conv2D('final_upconv', inputs, 2, [1,1], [1,1], 'SAME')

        return up_conv_f