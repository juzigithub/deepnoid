'''
model construct

by 신형은 주임
'''
import tensorflow as tf
import utils
# import config as cfg


class Model:
    def __init__(self, loss='dice', upsampling_mode='resize', normalization_mode='Batch', groupnormalization_n=2, downsampling_option='neighbor',
                 model_root_channel=32, img_size=256, batch_size=20, n_channel=1, n_class=2, activation='relu', channel_mode='NHWC'):

        #######################여기를 없애시면 됩니다.#################################
        self.batch_size = batch_size
        self.normalization_mode = normalization_mode
        self.model_channel = model_root_channel
        self.upsampling_mode = upsampling_mode
        self.loss_mode = loss
        self.group_n = groupnormalization_n
        self.activation = activation
        self.channel_mode = channel_mode
        self.downsampling_opt = downsampling_option
        self.img_size = img_size
        ###########################################################################

        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        self.training = tf.placeholder(tf.bool, name='training')
        self.X = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, n_channel], name='X')
        self.Y = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, n_class], name='Y')
        self.X_ADD = tf.placeholder(tf.int32, [None, 3], name='X_ADD')

        # iterator 설정
        self.dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y, self.X_ADD)).shuffle(buffer_size=3000) # buffer_size도 cfg에 있습니다.
        self.dataset = self.dataset.batch(self.batch_size).repeat()
        # self.features, self.labels, self.address 를 iterator 변수로 설정 -> iterator 변수 호출 시 다음 데이터를 불러옵니다.
        self.iter = self.dataset.make_initializable_iterator()
        self.features, self.labels, self.address = self.iter.get_next()  # 기존 self.X, self.Y 들어갈 자리에 self.features, self.labels 입력

        self.logit = self.u_net()

        self.pred = tf.nn.softmax(logits=self.logit)

        self.foreground_predicted, self.background_predicted = tf.split(self.pred, [1, 1], 3)

        self.foreground_truth, self.background_truth = tf.split(self.labels, [1, 1], 3)

        ################### utils에 loss 함수를 선택할 수 있는 함수를 만들고, cfg.LOSS_FUNC 에 입력한 loss 함수를 불러오도록 합니다. ###################
        self.loss = utils.cross_entropy(output=self.foreground_predicted, target=self.foreground_truth)
        self.results = list(utils.iou_coe(output=self.foreground_predicted, target=self.foreground_truth))


    def u_net(self):
        depth = 4
        up_conv = [0] * depth
        up_deconv = [0] * depth
        up_norm = [0] * depth
        up_act = [0] * depth
        up_concat = [0] * depth
        down_conv = [0] * depth
        down_norm = [0] * depth
        down_act = [0] * depth
        down_pool = [0] * depth

        with tf.variable_scope('down'):

            inputs = self.features     # iterator 변수 self.features 를 이용해 inputs 생성
            channel_n = self.model_channel
            pool_size = self.img_size

            ########## 처음 코드 실행하면 모델 구조 출력되도록 해볼까요? #########
            for i in range(depth):

                down_conv[i] = utils.conv2D(str(i) + '_conv1', inputs, channel_n, [3,3], [1,1], padding = 'SAME')
                down_norm[i] = utils.Normalization(down_conv[i], 'batch',self.training, str(i) + '_norm1')
                down_act[i] = utils.activation(str(i) + '_act1', down_norm[i], 'relu')
                down_conv[i] = utils.conv2D(str(i) + '_conv2', down_act[i], channel_n, [3,3], [1,1], 'SAME')
                down_norm[i] = utils.Normalization(down_conv[i], 'batch',self.training, str(i) + '_norm2')
                down_act[i] = utils.activation(str(i) + '_act2', down_norm[i], 'relu')
                down_pool[i] = utils.maxpool(str(i) + '_pool1', down_act[i], [2,2], [2,2], 'SAME')

                inputs = down_pool[i]

                channel_n *= 2
                pool_size //= 2

            down_conv_f = utils.conv2D('final_conv1', inputs, channel_n, [3, 3], [1, 1], padding='SAME')
            down_norm_f = utils.Normalization(down_conv_f, 'batch', self.training, 'final_norm1')
            down_act_f = utils.activation('final_conv1', down_norm_f, 'relu')
            down_conv_f = utils.conv2D('final_conv2', down_act_f, channel_n, [3, 3], [1, 1], padding='SAME')
            down_norm_f= utils.Normalization(down_conv_f, 'batch', self.training, 'final_norm2')
            down_act_f = utils.activation('final_conv2', down_norm_f, 'relu')

        with tf.variable_scope('up'):

            inputs = down_act_f

            for i in reversed(range(depth)):

                pool_size *=2

                up_deconv[i] = utils.deconv2D(str(i) +'_upconv1', inputs, [3,3,channel_n //2, channel_n],
                                              [-1, pool_size, pool_size, channel_n // 2], [1, 2, 2, 1], 'SAME')
                up_deconv[i] = tf.reshape(up_deconv[i], shape=[-1, pool_size, pool_size, channel_n // 2])

                up_norm[i] = utils.Normalization(up_deconv[i], 'batch', self.training, str(i) + '_upnorm1')
                up_act[i] = utils.activation(str(i) + '_upact1', up_norm[i], 'relu')
                up_concat[i] = utils.concat(str(i) + '_upconcat1', [up_act[i], down_act[i]], 3)

                channel_n //= 2

                up_conv[i] = utils.conv2D(str(i) + '_upconv1', up_concat[i], channel_n, [3,3], [1,1], 'SAME')
                # up_norm[i] = utils.Normalization(up_conv[i], 'batch', self.training, str(i) + '_upnorm1')
                up_act[i] = utils.activation(str(i) + '_upact1', up_conv[i], 'relu')
                up_conv[i] = utils.conv2D(str(i) + '_upconv2', up_act[i], channel_n, [3,3], [1,1], 'SAME')
                up_norm[i] = utils.Normalization(up_conv[i], 'batch', self.training, str(i) + '_upnorm2')
                up_act[i] = utils.activation(str(i) + '_upact2', up_norm[i], 'relu')

                inputs = up_act[i]

            up_conv_f = utils.conv2D('final_upconv1', inputs, 2, [1,1], [1,1], 'SAME')

        return up_conv_f


if __name__ == '__main__':
    ###### 최종적으로 train_mobile.py 에서 Model()을 불러올 때 model = Model(안에 아무것도 들어가지 않도록) 코드를 짜시면 됩니다.########
    # model = Model()
    model = Model(img_size=256,
                  n_channel=1,
                  n_class=2,
                  batch_size=28)










