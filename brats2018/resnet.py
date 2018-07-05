
import tensorflow as tf
import utils
import config as cfg


class Model:
    def __init__(self):

        self.drop_rate = tf.placeholder(tf.float32, name='drop_rate')
        self.training = tf.placeholder(tf.bool, name='training')
        # self.X = tf.placeholder(tf.float32, [None, cfg.IMG_SIZE, cfg.IMG_SIZE, 4], name='X')
        # self.Y = tf.placeholder(tf.float32, [None, cfg.IMG_SIZE, cfg.IMG_SIZE, 1], name='Y')
        self.X = tf.placeholder(tf.float32, [None, cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.RGB_CHANNEL], name='X')
        self.Y = tf.placeholder(tf.float32, [None, cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.N_CLASS], name='Y')

        self.logit = self.resnet

        self.pred = tf.nn.softmax(logits=self.logit)
        # 활성화 시킨 probability map을 split 하여 foreground와 background로 분리합니다.

        self.foreground_predicted, self.background_truth = tf.split(self.pred, [1, 1], 3)

        # 라벨이미지 역시 foreground와 background로 분리합니다
        self.foreground_truth, self.background_truth = tf.split(self.Y, [1, 1], 3)


        ################### utils에 loss 함수를 선택할 수 있는 함수를 만들고, cfg.LOSS_FUNC 에 입력한 loss 함수를 불러오도록 합니다. ###################
        self.loss = utils.select_loss(mode=cfg.LOSS_FUNC, output=self.foreground_predicted,
                                      target=self.foreground_truth)
        # self.loss = utils.dice_loss(output=self.foreground_predicted, target=self.foreground_truth)

        self.results = list(utils.iou_coe(output=self.foreground_predicted, target=self.foreground_truth))

    def resnet(self):

        inputs = self.X
        filters = cfg.INIT_FILTER

        print('Selected Mode : ', cfg.MODE)

        print('inputs : ' + str(inputs.shape))      # (?,256,256,1)

        # initial convolutional layer with normalization and activation.
        conv1 = utils.conv2D('init_conv1', inputs, filters, [7, 7], [1, 1], padding='SAME')
        print('conv1 : ' + str(conv1.shape))
        batch1 = utils.Normalization(conv1, 'batch', cfg.training, 'init_batch1')
        act1 = utils.activation('init_act1', batch1, cfg.ACTIVATION_FUNC)

        # maxpool to make output size 128 x 128
        pool1 = utils.maxpool('init_pool1', act1, [3, 3], [2, 2], 'SAME')
        # pool1 = utils.GlobalAveragePooling2D(act1, cfg.N_CLASS, 'init_pool1', keep_dims=True)

        print('pool1 : ' + str(pool1.shape))

        inputs = pool1

        # # chose among bottleneck_v1, residual_v1, bottleneck_v2, residual_v2
        # result, final_filter = select_resnet(cfg.MODE, inputs)

        # chose among bottleneck_v1_with_unet, bottleneck_v2_with_unet, residual_v1_with_unet, residual_v2_with_unet
        result = select_mode(cfg.MODE, inputs)

        return result

'''
        # # code below is just for testing blocks
        #
        # # initial filter numbers, 64
        # channel_n = cfg.INIT_FILTER
        # pool_size = cfg.IMG_SIZE
        #
        # # Method below pool img size at once.
        # # will be modified later.
        # stride_factor = pool_size // int(result.shape[1])
        # print('stride_Factror : ', stride_factor)
        # # [1, stride_factor, stride_factor, 1]
        #
        # up_deconv = utils.deconv2D('updeconv1', result, [3, 3, channel_n, final_filter],
        #                            [-1, pool_size, pool_size, channel_n], [1, stride_factor, stride_factor, 1], 'SAME')
        # up_deconv = tf.reshape(up_deconv, shape=[-1, pool_size, pool_size, channel_n])
        # up_norm = utils.Normalization(up_deconv, 'batch', cfg.training, 'upnorm1')
        # up_act = utils.activation('upact1', up_norm, cfg.ACTIVATION_FUNC)
        # result = utils.conv2D('final_upconv', up_act, 2, [1, 1], [1, 1], 'SAME')
'''

################################# Resnet blocks ###################################

def select_block(mode, inputs):
    if mode == 'bottleneck_block_v1':
        return bottleneck_block_v1(inputs)
    elif mode == 'bottleneck_block_v2':
        return bottleneck_block_v2(inputs)
    elif mode == 'residual_block_v1':
        return residual_block_v1(inputs)
    elif mode == 'residual_block_v2':
        return residual_block_v2(inputs)
    else:
        print("Not supported block. Select residual_block_v1, bottleneck_block_v1,residual_block_v2, bottleneck_block_v2")

def bottleneck_block_v1(inputs, filters, idx, training, strides, act_func, norm_type, group_n):

    shortcut = tf.identity(inputs, 'shortcut')
    shortcut = utils.conv2D(str(idx) + '_short_cut', shortcut, filters * 4, [1, 1], strides, padding='SAME')
    shortcut = utils.Normalization(shortcut, norm_type, training, str(idx) + '_short_cut_bn', G = group_n)

    conv = utils.conv2D(str(idx) + '_conv1', inputs, filters, [1, 1], [1, 1], padding='SAME')
    norm = utils.Normalization(conv, norm_type, training, str(idx) + '_norm1', G = group_n)
    act = utils.activation(str(idx) + '_act1', norm,act_func)
    conv = utils.conv2D(str(idx) + '_conv2', act, filters, [3, 3], strides, padding='SAME')
    norm = utils.Normalization(conv, norm_type, training, str(idx) + '_norm2', G = group_n)
    act = utils.activation(str(idx) + '_act2', norm, act_func)
    conv = utils.conv2D(str(idx) + '_conv3', act, filters * 4, [1, 1], [1, 1],
                           padding='SAME')
    norm = utils.Normalization(conv, norm_type, training, str(idx) + '_norm3', G = group_n)
    norm += shortcut
    act = utils.activation(str(idx) + '_act3', norm, act_func)

    return act

def bottleneck_block_v2(inputs, filters, idx, training, strides, act_func, norm_type, group_n):

    shortcut = tf.identity(inputs, 'shortcut')
    shortcut = utils.Normalization(shortcut, norm_type, training, str(idx) + '_short_cut_bn', G = group_n)
    shortcut = utils.activation(str(idx) + 'sc_act1', shortcut, act_func)
    shortcut = utils.conv2D(str(idx) + '_short_cut', shortcut, filters*4, [1, 1], strides, padding='SAME')

    conv = utils.conv2D(str(idx) + '_conv1', inputs, filters, [1, 1], [1, 1], padding='SAME')
    norm = utils.Normalization(conv, norm_type, training, str(idx) + '_norm1', G = group_n)
    act = utils.activation(str(idx) + '_act1', norm, act_func)
    conv = utils.conv2D(str(idx) + '_conv2', act, filters, [3, 3], strides, padding='SAME')
    norm = utils.Normalization(conv, norm_type, training, str(idx) + '_norm2', G = group_n)
    act = utils.activation(str(idx) + '_act2', norm, act_func)
    conv = utils.conv2D(str(idx) + '_conv3', act, filters * 4, [1, 1], [1, 1], padding='SAME')

    add = conv + shortcut
    return add

def residual_block_v1(inputs, filters, idx, training, strides, act_func, norm_type, group_n):

    shortcut = tf.identity(inputs, 'shortcut')
    shortcut = utils.conv2D(str(idx) + '_short_cut', shortcut, filters, [1, 1], strides, padding='SAME')
    shortcut = utils.Normalization(shortcut, norm_type, training, str(idx) + '_short_cut_bn', G = group_n)

    conv = utils.conv2D(str(idx) + '_conv1', inputs, filters, [3, 3], [1, 1], padding='SAME')
    norm = utils.Normalization(conv, norm_type, training, str(idx) + '_norm1', G = group_n)
    act = utils.activation(str(idx) + '_act1', norm, act_func)
    conv = utils.conv2D(str(idx) + '_conv2', act, filters, [3, 3], strides, padding='SAME')
    norm = utils.Normalization(conv, norm_type, training, str(idx) + '_norm2', G = group_n)
    norm += shortcut
    act = utils.activation(str(idx) + '_act2', norm, act_func)

    return act

def residual_block_v2(inputs, filters, idx, training, strides, act_func, norm_type, group_n):

    shortcut = tf.identity(inputs, 'shortcut')
    shortcut = utils.Normalization(shortcut, norm_type, training, str(idx) + '_short_cut_bn', G = group_n)
    shortcut = utils.activation(str(idx) + 'sc_act1', shortcut, act_func)
    shortcut = utils.conv2D(str(idx) + '_short_cut', shortcut, filters, [1, 1], strides, padding='SAME')

    conv = utils.conv2D(str(idx) + '_conv1', inputs, filters, [3, 3], [1, 1], padding='SAME')
    norm = utils.Normalization(conv, norm_type, training, str(idx) + '_norm1', G = group_n)
    act = utils.activation(str(idx) + '_act1', norm, act_func)
    conv = utils.conv2D(str(idx) + '_conv2', act, filters, [3, 3], strides, padding='SAME')

    add = conv + shortcut

    return add



################################### Resnet models #######################################

# can load resnet_50, resnet101 etc by setting cfg.n_filter_chunks

def select_resnet(mode, inputs):
    if mode == 'bottleneck_v1':
        return bottleneck_v1(inputs)
    elif mode == 'bottleneck_v2':
        return bottleneck_v2(inputs)
    elif mode == 'residual_v1':
        return residual_v1(inputs)
    elif mode == 'residual_v2':
        return residual_v2(inputs)
    else:
        print("Not supported block. Select residual_block_v1, bottleneck_block_v1,residual_block_v2, bottleneck_block_v2")

def bottleneck_v1(inputs):


    filter_chunk = cfg.n_filter_chunks

    n_filters = []
    filters = cfg.INIT_FILTER
    for i in range(cfg.RES_DEPTH):
        if i > 0:
            filters *= 2
        else:
            pass
        n_filters.append(filters)

    print('n_filters : ', n_filters)


    conv = [0] * cfg.v2_depth
    norm = [0] * cfg.v2_depth
    act = [0] * cfg.v2_depth
    middle = [0] * cfg.v2_depth
    shortcut = [0] * cfg.v2_depth

    # filters = int(filter_chunk[i])
    filters = cfg.INIT_FILTER

    for n in range(len(filter_chunk)):

        iterates = cfg.n_filter_chunks[n]
        print('iterates : ', iterates)

        # if n > 0:
        #     filters *= 2

        for i in range(int(iterates)):

            # if it's first loop, don't need to shrink size because maxpool would do.
            if i == iterates - 1 :
                strides = [2,2]
            else:
                strides = [1,1]

            shortcut[i] = tf.identity(inputs, 'shortcut')
            shortcut[i] = utils.conv2D(str(n) + str(i) + '_short_cut', shortcut[i], filters, [1, 1], strides, padding='SAME')
            shortcut[i] = utils.Normalization(shortcut[i], 'batch', cfg.training, str(n) + str(i) + '_short_cut_bn')

            print('{2} |shortcut filter{0} : {1}'.format(str(i), filters, str(n)))

            conv[i] = utils.conv2D(str(n) + str(i) + '_conv1', inputs, filters, [3, 3], [1, 1], padding='SAME')
            norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(n) + str(i) + '_norm1')
            act[i] = utils.activation(str(n) + str(i) + '_act1', norm[i], cfg.ACTIVATION_FUNC)
            conv[i] = utils.conv2D(str(n) + str(i) + '_conv2', act[i], filters, [3, 3], strides, padding='SAME')
            norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(n) + str(i) + '_norm2')
            norm[i] += shortcut[i]
            act[i] = utils.activation(str(n) + str(i) + '_act2', norm[i], cfg.ACTIVATION_FUNC)

            inputs = act[i]
            print('{2} |conv filter{0} : {1}'.format(str(i), filters, str(n)))
            middle[n] = act[int(iterates)-1]

        filters *= 2

    # print('{3} | conv[{0}] : {1}, shortcut : {2}'.format(str(i), conv[i].shape, shortcut.shape, str(n)))

    # inputs = conv[i] + shortcut
    final_filter = filters

    # up_inputs = conv[len(filter_chunk)-1] + shortcut
    up_inputs = middle[len(filter_chunk)-1]

    final_size = int(up_inputs.shape[1])
    print('final_size : ', final_size)
    # middle[n] = conv[i] + shortcut
    print('middle[{0}] : {1}'.format(str(n), middle[n].shape))

    return up_inputs, final_filter

def bottleneck_v2(inputs):

    filter_chunk = cfg.n_filter_chunks

    n_filters = []
    filters = cfg.INIT_FILTER

    print('n_filters : ', n_filters)

    print('filter_chunk : ', filter_chunk)

    conv = [0] * cfg.v2_depth
    norm = [0] * cfg.v2_depth
    act = [0] * cfg.v2_depth
    middle = [0] * cfg.v2_depth
    shortcut = [0] * cfg.v2_depth

    # filters = int(filter_chunk[i])
    filters = cfg.INIT_FILTER

    for n in range(len(filter_chunk)):

        iterates = cfg.n_filter_chunks[n]
        print('iterates : ', iterates)

        if n > 0:
            filters *= 2

        for i in range(int(iterates)):

            # if it's first loop, don't need to shrink size because maxpool would do.
            if i == iterates - 1 :
                strides = [2,2]
            else:
                strides = [1,1]

            shortcut[i] = tf.identity(inputs, 'shortcut')
            shortcut[i] = utils.Normalization(shortcut[i], 'batch', cfg.training, str(n) + str(i) + '_short_cut_bn')
            shortcut[i] = utils.activation(str(n) + str(i) + 'sc_act1', shortcut[i], cfg.ACTIVATION_FUNC)
            shortcut[i] = utils.conv2D(str(n) + str(i) + '_short_cut', shortcut[i], filters*4, [1, 1], strides, padding='SAME')

            print('{2} |shortcut filter{0} : {1}'.format(str(i), filters, str(n)))

            conv[i] = utils.conv2D(str(n) + str(i) + '_conv1', inputs, filters, [1, 1], [1, 1], padding='SAME')
            norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(n) + str(i) + '_norm1')
            act[i] = utils.activation(str(n) + str(i) + '_act1', norm[i], cfg.ACTIVATION_FUNC)
            conv[i] = utils.conv2D(str(n) + str(i) + '_conv2', act[i], filters, [3, 3], strides, padding='SAME')
            norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(n) + str(i) + '_norm2')
            act[i] = utils.activation(str(n) + str(i) + '_act2', norm[i], cfg.ACTIVATION_FUNC)
            conv[i] = utils.conv2D(str(n) + str(i) + '_conv3', act[i], filters * 4, [1, 1], [1, 1],
                                   padding='SAME')
            inputs = conv[i] + shortcut[i]
            print('{2} |conv filter{0} : {1}'.format(str(i), filters, str(n)))
            middle[n] = conv[int(iterates)-1] + shortcut[int(iterates)-1]

    # print('{3} | conv[{0}] : {1}, shortcut : {2}'.format(str(i), conv[i].shape, shortcut.shape, str(n)))

    # inputs = conv[i] + shortcut
    final_filter = filters*4

    # up_inputs = conv[len(filter_chunk)-1] + shortcut
    up_inputs = middle[len(filter_chunk)-1]

    final_size = int(up_inputs.shape[1])
    print('final_size : ', final_size)
    # middle[n] = conv[i] + shortcut
    print('middle[{0}] : {1}'.format(str(n), middle[n].shape))

    print(inputs.shape)

    return up_inputs, final_filter

def residual_v1(inputs):

    filter_chunk = []
    filters = cfg.INIT_FILTER
    for i in range(cfg.RES_DEPTH):
        if i > 0:
            filters *= 2
        else:
            pass
        filter_chunk.append(filters)

    print('filter_chunk : ', filter_chunk)

    conv = [0] * len(filter_chunk)
    norm = [0] * len(filter_chunk)
    act = [0] * len(filter_chunk)

    for i in range(len(filter_chunk)):

        filters = int(filter_chunk[i])

        if i > 0 :
            strides = [2,2]
        else:
            strides = [1,1]

        shortcut = tf.identity(inputs, 'shortcut')
        shortcut = utils.conv2D(str(i) + '_short_cut', shortcut, filters, [1, 1], strides, padding='SAME')
        shortcut = utils.Normalization(shortcut, 'batch', cfg.training, str(i) + '_short_cut_bn')

        conv[i] = utils.conv2D(str(i) + '_conv1', inputs, filters, [3, 3], [1, 1], padding='SAME')
        norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(i) + '_norm1')
        act[i] = utils.activation(str(i) + '_act1', norm[i], cfg.ACTIVATION_FUNC)
        conv[i] = utils.conv2D(str(i) + '_conv2', act[i], filters, [3, 3], strides, padding='SAME')
        norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(i) + '_norm2')
        norm[i] += shortcut
        act[i] = utils.activation(str(i) + '_act2', norm[i], cfg.ACTIVATION_FUNC)

        print('conv[{0}] : {1}, shortcut : {2}'.format(str(i), conv[i].shape, shortcut.shape))

        final_filter = filter_chunk[len(filter_chunk) - 1]

        inputs = act[i]

        print(inputs.shape)

    return inputs, final_filter

def residual_v2(inputs):


    filter_chunk = cfg.n_filter_chunks

    n_filters = []
    filters = cfg.INIT_FILTER
    for i in range(cfg.RES_DEPTH):
        if i > 0:
            filters *= 2
        else:
            pass
        n_filters.append(filters)

    print('n_filters : ', n_filters)


    conv = [0] * cfg.v2_depth
    norm = [0] * cfg.v2_depth
    act = [0] * cfg.v2_depth
    middle = [0] * cfg.v2_depth
    shortcut = [0] * cfg.v2_depth

    # filters = int(filter_chunk[i])
    filters = cfg.INIT_FILTER

    for n in range(len(filter_chunk)):

        iterates = cfg.n_filter_chunks[n]
        print('iterates : ', iterates)

        if n > 0:
            filters *= 2

        for i in range(int(iterates)):

            # if it's first loop, don't need to shrink size because maxpool would do.
            if i == iterates - 1 :
                strides = [2,2]
            else:
                strides = [1,1]

            shortcut[i] = tf.identity(inputs, 'shortcut')
            shortcut[i] = utils.Normalization(shortcut[i], 'batch', cfg.training, str(n) + str(i) + '_short_cut_bn')
            shortcut[i] = utils.activation(str(n) + str(i) + 'sc_act1', shortcut[i], cfg.ACTIVATION_FUNC)
            shortcut[i] = utils.conv2D(str(n) + str(i) + '_short_cut', shortcut[i], filters , [1, 1], strides,
                                       padding='SAME')

            print('{2} |shortcut filter{0} : {1}'.format(str(i), filters, str(n)))

            conv[i] = utils.conv2D(str(n) + str(i) + '_conv1', inputs, filters, [3, 3], [1, 1], padding='SAME')
            norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(n) + str(i) + '_norm1')
            act[i] = utils.activation(str(n) + str(i) + '_act1', norm[i], cfg.ACTIVATION_FUNC)
            conv[i] = utils.conv2D(str(n) + str(i) + '_conv2', act[i], filters, [3, 3], strides, padding='SAME')

            inputs = conv[i] + shortcut[i]
            print('{2} |conv filter{0} : {1}'.format(str(i), filters, str(n)))
            middle[n] = conv[int(iterates) - 1] + shortcut[int(iterates) - 1]

    # print('{3} | conv[{0}] : {1}, shortcut : {2}'.format(str(i), conv[i].shape, shortcut.shape, str(n)))

    # inputs = conv[i] + shortcut
    final_filter = filters

    # up_inputs = conv[len(filter_chunk)-1] + shortcut
    up_inputs = middle[len(filter_chunk) - 1]

    final_size = int(up_inputs.shape[1])
    print('final_size : ', final_size)
    # middle[n] = conv[i] + shortcut
    print('middle[{0}] : {1}'.format(str(n), middle[n].shape))


    return inputs, final_filter

'''
conv = [0] * len(conv_filter)
norm = [0] * len(conv_filter)
act = [0] * len(conv_filter)

for i in range(len(conv_filter)):

    filters = int(conv_filter[i])

    if i > 0 :
        strides = [2,2]
    else:
        strides = [1,1]

    shortcut = tf.identity(inputs, 'shortcut')
    shortcut = utils.Normalization(shortcut, 'batch', cfg.training, str(i) + '_short_cut_bn')
    shortcut = utils.activation(str(i) + 'sc_act1', shortcut, cfg.ACTIVATION_FUNC)
    shortcut = utils.conv2D(str(i) + '_short_cut', shortcut, filters, [1, 1], strides, padding='SAME')
    shortcut = utils.Normalization(shortcut, 'batch', cfg.training, str(i) + '_short_cut_bn2')
    shortcut = utils.activation(str(i) + 'sc_act2', shortcut, cfg.ACTIVATION_FUNC)
    shortcut = utils.conv2D(str(i) + '_short_cut2', shortcut, filters, [1, 1], [1,1], padding='SAME')

    conv[i] = utils.conv2D(str(i) + '_conv1', inputs, filters, [3, 3], [1, 1], padding='SAME')
    norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(i) + '_norm1')
    act[i] = utils.activation(str(i) + '_act1', norm[i], cfg.ACTIVATION_FUNC)
    conv[i] = utils.conv2D(str(i) + '_conv2', act[i], filters, [3, 3], strides, padding='SAME')

    print('conv[{0}] : {1}, shortcut : {2}'.format(str(i), conv[i].shape, shortcut.shape))

    inputs = conv[i] + shortcut

    final_filter = conv_filter[len(conv_filter)-2]
'''

############################## Resnet models with unet upsampling ##########################################

def select_mode(mode, inputs):
    if mode == 'bottleneck_v1_with_unet':
        return bottleneck_v1_with_unet(inputs)
    elif mode == 'bottleneck_v2_with_unet':
        return bottleneck_v2_with_unet(inputs)
    elif mode == 'residual_v1_with_unet':
        return residual_v1_with_unet(inputs)
    elif mode == 'residual_v2_with_unet':
        return residual_v2_with_unet(inputs)
    else:
        print("Not supported block. Select bottleneck_v1, bottleneck_v2,residual_v1, residual_v2")

def bottleneck_v1_with_unet(inputs):


    ################# Resnet bottleneck v2 with unet upsample #########################

    filter_chunk = cfg.n_filter_chunks

    n_filters = []
    filters = cfg.INIT_FILTER
    for i in range(cfg.RES_DEPTH):
        if i > 0:
            filters *= 2
        else:
            pass
        n_filters.append(filters)

    print('n_filters : ', n_filters)


    conv = [0] * cfg.v2_depth
    norm = [0] * cfg.v2_depth
    act = [0] * cfg.v2_depth
    middle = [0] * cfg.v2_depth
    up_concat = [0] * cfg.v2_depth
    up_conv = [0] * cfg.v2_depth
    up_conv2 = [0] * cfg.v2_depth
    up_deconv = [0] * cfg.v2_depth
    up_norm = [0] * cfg.v2_depth
    up_act = [0] * cfg.v2_depth
    shortcut = [0] * cfg.v2_depth

    # filters = int(filter_chunk[i])
    filters = cfg.INIT_FILTER

    for n in range(len(filter_chunk)):

        iterates = cfg.n_filter_chunks[n]
        print('iterates : ', iterates)

        # if n > 0:
        #     filters *= 2

        for i in range(int(iterates)):

            # if it's first loop, don't need to shrink size because maxpool would do.
            if i == iterates - 1 :
                strides = [2,2]
            else:
                strides = [1,1]

            shortcut[i] = tf.identity(inputs, 'shortcut')
            shortcut[i] = utils.conv2D(str(n) + str(i) + '_short_cut', shortcut[i], filters * 4, [1, 1], strides, padding='SAME')
            shortcut[i] = utils.Normalization(shortcut[i], 'batch', cfg.training, str(n) + str(i) + '_short_cut_bn')

            print('{2} |shortcut filter{0} : {1}'.format(str(i), filters, str(n)))

            conv[i] = utils.conv2D(str(n) + str(i) + '_conv1', inputs, filters, [1, 1], [1, 1], padding='SAME')
            norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(n) + str(i) + '_norm1')
            act[i] = utils.activation(str(n) + str(i) + '_act1', norm[i], cfg.ACTIVATION_FUNC)
            conv[i] = utils.conv2D(str(n) + str(i) + '_conv2', act[i], filters, [3, 3], strides, padding='SAME')
            norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(n) + str(i) + '_norm2')
            act[i] = utils.activation(str(n) + str(i) + '_act2', norm[i], cfg.ACTIVATION_FUNC)
            conv[i] = utils.conv2D(str(n) + str(i) + '_conv3', act[i], filters * 4, [1, 1], [1, 1],
                                   padding='SAME')
            norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(n) + str(i) + '_norm3')
            norm[i] += shortcut[i]
            act[i] = utils.activation(str(n) + str(i) + '_act3', norm[i], cfg.ACTIVATION_FUNC)

            inputs = act[i]
            print('{2} |conv filter{0} : {1}'.format(str(i), filters, str(n)))
            middle[n] = act[int(iterates)-1]

        filters *= 2

    # print('{3} | conv[{0}] : {1}, shortcut : {2}'.format(str(i), conv[i].shape, shortcut.shape, str(n)))

    # inputs = conv[i] + shortcut
    final_filter = filter_chunk[len(filter_chunk)-1]*4

    # up_inputs = conv[len(filter_chunk)-1] + shortcut
    up_inputs = middle[len(filter_chunk)-1]

    final_size = int(up_inputs.shape[1])
    print('final_size : ', final_size)
    # middle[n] = conv[i] + shortcut
    print('middle[{0}] : {1}'.format(str(n), middle[n].shape))

    norm = utils.Normalization(up_inputs, 'batch', cfg.training, 'norm1')
    act = utils.activation('act1', norm, cfg.ACTIVATION_FUNC)
    up_inputs= utils.conv2D('conv2', act, filters*4, [3, 3], [2,2], padding='SAME')

    ############## up_sample #############

    filters *= 4
    print('filters : ', filters)

    for n in reversed(range(len(filter_chunk))):

        print('filter2 : ', filters)

        up_deconv[n] = utils.deconv2D(str(n) + '_updeconv1', up_inputs, [3, 3, filters//2,  filters],
                                      [-1, final_size, final_size, filters//2,], [1, 2, 2, 1], 'SAME')
        up_deconv[n] = tf.reshape(up_deconv[n], shape=[-1, final_size, final_size, filters//2,])

        up_norm[n] = utils.Normalization(up_deconv[n], 'batch', cfg.training, str(n) + '_upnorm1')
        up_act[n] = utils.activation(str(n) + '_upact1', up_norm[n], cfg.ACTIVATION_FUNC)
        up_concat[n] = utils.concat(str(n) + '_upconcat1', [up_act[n], middle[n]], 3)

        filters //= 2
        final_size *= 2

        up_conv[n] = utils.conv2D(str(n) + '_upconv1', up_concat[n], filters, [3, 3], [1, 1], 'SAME')
        up_norm[n] = utils.Normalization(up_conv[n], 'batch', cfg.training, str(n) + '_upnorm2')
        up_act[n] = utils.activation(str(n) + '_upact1', up_norm[n], cfg.ACTIVATION_FUNC)
        up_conv2[n] = utils.conv2D(str(n) + '_upconv2', up_act[n], filters, [3, 3], [1, 1], 'SAME')
        up_norm[n] = utils.Normalization(up_conv2[n], 'batch', cfg.training, str(n) + '_upnorm3')
        up_act[n] = utils.activation(str(n) + '_upact2', up_norm[n], cfg.ACTIVATION_FUNC)
        print('{0}_updeconv1 : {1}, {0}_upconcat1 : {2}, {0}_upconv1 : {3}, {0}_upconv2 : {4}'
              .format(str(n), up_deconv[n].shape, up_concat[n].shape, up_conv[n].shape, up_conv2[n].shape))

        up_inputs = up_act[n]

    up_deconv = utils.deconv2D('final_updeconv1', up_inputs, [3, 3, filters // 2, filters],
                                  [-1, final_size, final_size, filters // 2, ], [1, 2, 2, 1], 'SAME')
    up_deconv = tf.reshape(up_deconv, shape=[-1, final_size, final_size, filters // 2, ])
    up_norm = utils.Normalization(up_deconv, 'batch', cfg.training, 'final_upnorm1')
    up_act = utils.activation('final_upact1', up_norm, cfg.ACTIVATION_FUNC)
    up_deconv = utils.deconv2D('final_updeconv2', up_act, [3, 3, filters // 4, filters//2],
                                  [-1, final_size*2, final_size*2, filters // 4, ], [1, 2, 2, 1], 'SAME')
    up_deconv = tf.reshape(up_deconv, shape=[-1, final_size*2, final_size*2, filters // 4, ])
    up_norm = utils.Normalization(up_deconv, 'batch', cfg.training, 'final_upnorm2')
    up_act = utils.activation('final_upact2', up_norm, cfg.ACTIVATION_FUNC)

    up_conv_f = utils.conv2D('final_upconv1', up_act, 2, [1, 1], [1, 1], 'SAME')
    print('final_upconv1 : ', up_conv_f.shape)

    return up_conv_f

def bottleneck_v2_with_unet(inputs):

    ################# Resnet bottleneck v2 with unet upsample #########################

    filter_chunk = cfg.n_filter_chunks

    n_filters = []
    filters = cfg.INIT_FILTER
    for i in range(cfg.RES_DEPTH):
        if i > 0:
            filters *= 2
        else:
            pass
        n_filters.append(filters)

    print('n_filters : ', n_filters)


    conv = [0] * cfg.v2_depth
    norm = [0] * cfg.v2_depth
    act = [0] * cfg.v2_depth
    middle = [0] * cfg.v2_depth
    up_concat = [0] * cfg.v2_depth
    up_conv = [0] * cfg.v2_depth
    up_conv2 = [0] * cfg.v2_depth
    up_deconv = [0] * cfg.v2_depth
    up_norm = [0] * cfg.v2_depth
    up_act = [0] * cfg.v2_depth
    shortcut = [0] * cfg.v2_depth

    # filters = int(filter_chunk[i])
    filters = cfg.INIT_FILTER

    for n in range(len(filter_chunk)):

        iterates = cfg.n_filter_chunks[n]
        print('iterates : ', iterates)

        # if n > 0:
        #     filters *= 2

        for i in range(int(iterates)):

            # if it's first loop, don't need to shrink size because maxpool would do.
            if i == iterates - 1 :
                strides = [2,2]
            else:
                strides = [1,1]

            shortcut[i] = tf.identity(inputs, 'shortcut')
            shortcut[i] = utils.Normalization(shortcut[i], 'batch', cfg.training, str(n) + str(i) + '_short_cut_bn')
            shortcut[i] = utils.activation(str(n) + str(i) + 'sc_act1', shortcut[i], cfg.ACTIVATION_FUNC)
            shortcut[i] = utils.conv2D(str(n) + str(i) + '_short_cut', shortcut[i], filters*4, [1, 1], strides, padding='SAME')

            print('{2} |shortcut filter{0} : {1}'.format(str(i), filters, str(n)))

            conv[i] = utils.conv2D(str(n) + str(i) + '_conv1', inputs, filters, [1, 1], [1, 1], padding='SAME')
            norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(n) + str(i) + '_norm1')
            act[i] = utils.activation(str(n) + str(i) + '_act1', norm[i], cfg.ACTIVATION_FUNC)
            conv[i] = utils.conv2D(str(n) + str(i) + '_conv2', act[i], filters, [3, 3], strides, padding='SAME')
            norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(n) + str(i) + '_norm2')
            act[i] = utils.activation(str(n) + str(i) + '_act2', norm[i], cfg.ACTIVATION_FUNC)
            conv[i] = utils.conv2D(str(n) + str(i) + '_conv3', act[i], filters * 4, [1, 1], [1, 1],
                                   padding='SAME')
            inputs = conv[i] + shortcut[i]
            print('{2} |conv filter{0} : {1}'.format(str(i), filters, str(n)))
            middle[n] = conv[int(iterates)-1] + shortcut[int(iterates)-1]

        filters *= 2

    # print('{3} | conv[{0}] : {1}, shortcut : {2}'.format(str(i), conv[i].shape, shortcut.shape, str(n)))

    # inputs = conv[i] + shortcut
    final_filter = filter_chunk[len(filter_chunk)-1]*4

    # up_inputs = conv[len(filter_chunk)-1] + shortcut
    up_inputs = middle[len(filter_chunk)-1]

    final_size = int(up_inputs.shape[1])
    print('final_size : ', final_size)
    # middle[n] = conv[i] + shortcut
    print('middle[{0}] : {1}'.format(str(n), middle[n].shape))

    norm = utils.Normalization(up_inputs, 'batch', cfg.training, 'norm1')
    act = utils.activation('act1', norm, cfg.ACTIVATION_FUNC)
    up_inputs= utils.conv2D('conv2', act, filters*4, [3, 3], [2,2], padding='SAME')

    ############## up_sample #############

    filters *= 4
    print('filters : ', filters)

    for n in reversed(range(len(filter_chunk))):

        print('filter2 : ', filters)

        up_deconv[n] = utils.deconv2D(str(n) + '_updeconv1', up_inputs, [3, 3, filters//2,  filters],
                                      [-1, final_size, final_size, filters//2,], [1, 2, 2, 1], 'SAME')
        up_deconv[n] = tf.reshape(up_deconv[n], shape=[-1, final_size, final_size, filters//2,])

        up_norm[n] = utils.Normalization(up_deconv[n], 'batch', cfg.training, str(n) + '_upnorm1')
        up_act[n] = utils.activation(str(n) + '_upact1', up_norm[n], cfg.ACTIVATION_FUNC)
        up_concat[n] = utils.concat(str(n) + '_upconcat1', [up_act[n], middle[n]], 3)

        filters //= 2
        final_size *= 2

        up_conv[n] = utils.conv2D(str(n) + '_upconv1', up_concat[n], filters, [3, 3], [1, 1], 'SAME')
        up_norm[n] = utils.Normalization(up_conv[n], 'batch', cfg.training, str(n) + '_upnorm2')
        up_act[n] = utils.activation(str(n) + '_upact1', up_norm[n], cfg.ACTIVATION_FUNC)
        up_conv2[n] = utils.conv2D(str(n) + '_upconv2', up_act[n], filters, [3, 3], [1, 1], 'SAME')
        up_norm[n] = utils.Normalization(up_conv2[n], 'batch', cfg.training, str(n) + '_upnorm3')
        up_act[n] = utils.activation(str(n) + '_upact2', up_norm[n], cfg.ACTIVATION_FUNC)
        print('{0}_updeconv1 : {1}, {0}_upconcat1 : {2}, {0}_upconv1 : {3}, {0}_upconv2 : {4}'
              .format(str(n), up_deconv[n].shape, up_concat[n].shape, up_conv[n].shape, up_conv2[n].shape))

        up_inputs = up_act[n]

    up_deconv = utils.deconv2D('final_updeconv1', up_inputs, [3, 3, filters // 2, filters],
                                  [-1, final_size, final_size, filters // 2, ], [1, 2, 2, 1], 'SAME')
    up_deconv = tf.reshape(up_deconv, shape=[-1, final_size, final_size, filters // 2, ])
    up_norm = utils.Normalization(up_deconv, 'batch', cfg.training, 'final_upnorm1')
    up_act = utils.activation('final_upact1', up_norm, cfg.ACTIVATION_FUNC)
    up_deconv = utils.deconv2D('final_updeconv2', up_act, [3, 3, filters // 4, filters//2],
                                  [-1, final_size*2, final_size*2, filters // 4, ], [1, 2, 2, 1], 'SAME')
    up_deconv = tf.reshape(up_deconv, shape=[-1, final_size*2, final_size*2, filters // 4, ])
    up_norm = utils.Normalization(up_deconv, 'batch', cfg.training, 'final_upnorm2')
    up_act = utils.activation('final_upact2', up_norm, cfg.ACTIVATION_FUNC)

    up_conv_f = utils.conv2D('final_upconv1', up_act, 2, [1, 1], [1, 1], 'SAME')
    print('final_upconv1 : ', up_conv_f.shape)

    return up_conv_f

def residual_v1_with_unet(inputs):

    ################# Resnet v1 with unet upsample #########################

    filter_chunk = cfg.n_filter_chunks

    n_filters = []
    filters = cfg.INIT_FILTER

    print('n_filters : ', n_filters)


    conv = [0] * cfg.v2_depth
    norm = [0] * cfg.v2_depth
    act = [0] * cfg.v2_depth
    middle = [0] * cfg.v2_depth
    up_concat = [0] * cfg.v2_depth
    up_conv = [0] * cfg.v2_depth
    up_conv2 = [0] * cfg.v2_depth
    up_deconv = [0] * cfg.v2_depth
    up_norm = [0] * cfg.v2_depth
    up_act = [0] * cfg.v2_depth
    shortcut = [0] * cfg.v2_depth

    # filters = int(filter_chunk[i])
    filters = cfg.INIT_FILTER

    for n in range(len(filter_chunk)):

        iterates = cfg.n_filter_chunks[n]
        print('iterates : ', iterates)

        # if n > 0:
        #     filters *= 2

        for i in range(int(iterates)):

            # if it's first loop, don't need to shrink size because maxpool would do.
            if i == iterates - 1 :
                strides = [2,2]
            else:
                strides = [1,1]

            shortcut[i] = tf.identity(inputs, 'shortcut')
            shortcut[i] = utils.conv2D(str(n) + str(i) + '_short_cut', shortcut[i], filters, [1, 1], strides, padding='SAME')
            shortcut[i] = utils.Normalization(shortcut[i], 'batch', cfg.training, str(n) + str(i) + '_short_cut_bn')

            print('{2} |shortcut filter{0} : {1}'.format(str(i), filters, str(n)))

            conv[i] = utils.conv2D(str(n) + str(i) + '_conv1', inputs, filters, [3, 3], [1, 1], padding='SAME')
            norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(n) + str(i) + '_norm1')
            act[i] = utils.activation(str(n) + str(i) + '_act1', norm[i], cfg.ACTIVATION_FUNC)
            conv[i] = utils.conv2D(str(n) + str(i) + '_conv2', act[i], filters, [3, 3], strides, padding='SAME')
            norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(n) + str(i) + '_norm2')
            norm[i] += shortcut[i]
            act[i] = utils.activation(str(n) + str(i) + '_act2', norm[i], cfg.ACTIVATION_FUNC)

            inputs = act[i]
            print('{2} |conv filter{0} : {1}'.format(str(i), filters, str(n)))
            middle[n] = act[int(iterates)-1]

        filters *= 2

    # print('{3} | conv[{0}] : {1}, shortcut : {2}'.format(str(i), conv[i].shape, shortcut.shape, str(n)))

    # inputs = conv[i] + shortcut
    final_filter = filters

    # up_inputs = conv[len(filter_chunk)-1] + shortcut
    up_inputs = middle[len(filter_chunk)-1]

    final_size = int(up_inputs.shape[1])
    print('final_size : ', final_size)
    # middle[n] = conv[i] + shortcut
    print('middle[{0}] : {1}'.format(str(n), middle[n].shape))

    norm = utils.Normalization(up_inputs, 'batch', cfg.training, 'norm1')
    act = utils.activation('act1', norm, cfg.ACTIVATION_FUNC)
    up_inputs= utils.conv2D('conv2', act, filters, [3, 3], [2,2], padding='SAME')

    ############## up_sample #############

    print('filters : ', filters)

    for n in reversed(range(len(filter_chunk))):

        print('filter2 : ', filters)

        up_deconv[n] = utils.deconv2D(str(n) + '_updeconv1', up_inputs, [3, 3, filters//2,  filters],
                                      [-1, final_size, final_size, filters//2,], [1, 2, 2, 1], 'SAME')
        up_deconv[n] = tf.reshape(up_deconv[n], shape=[-1, final_size, final_size, filters//2,])

        up_norm[n] = utils.Normalization(up_deconv[n], 'batch', cfg.training, str(n) + '_upnorm1')
        up_act[n] = utils.activation(str(n) + '_upact1', up_norm[n], cfg.ACTIVATION_FUNC)
        up_concat[n] = utils.concat(str(n) + '_upconcat1', [up_act[n], middle[n]], 3)

        filters //= 2
        final_size *= 2

        up_conv[n] = utils.conv2D(str(n) + '_upconv1', up_concat[n], filters, [3, 3], [1, 1], 'SAME')
        up_norm[n] = utils.Normalization(up_conv[n], 'batch', cfg.training, str(n) + '_upnorm2')
        up_act[n] = utils.activation(str(n) + '_upact1', up_norm[n], cfg.ACTIVATION_FUNC)
        up_conv2[n] = utils.conv2D(str(n) + '_upconv2', up_act[n], filters, [3, 3], [1, 1], 'SAME')
        up_norm[n] = utils.Normalization(up_conv2[n], 'batch', cfg.training, str(n) + '_upnorm3')
        up_act[n] = utils.activation(str(n) + '_upact2', up_norm[n], cfg.ACTIVATION_FUNC)
        print('{0}_updeconv1 : {1}, {0}_upconcat1 : {2}, {0}_upconv1 : {3}, {0}_upconv2 : {4}'
              .format(str(n), up_deconv[n].shape, up_concat[n].shape, up_conv[n].shape, up_conv2[n].shape))

        up_inputs = up_act[n]

    up_deconv = utils.deconv2D('final_updeconv1', up_inputs, [3, 3, filters // 2, filters],
                                  [-1, final_size, final_size, filters // 2, ], [1, 2, 2, 1], 'SAME')
    up_deconv = tf.reshape(up_deconv, shape=[-1, final_size, final_size, filters // 2, ])
    up_norm = utils.Normalization(up_deconv, 'batch', cfg.training, 'final_upnorm1')
    up_act = utils.activation('final_upact1', up_norm, cfg.ACTIVATION_FUNC)
    up_deconv = utils.deconv2D('final_updeconv2', up_act, [3, 3, filters // 4, filters//2],
                                  [-1, final_size*2, final_size*2, filters // 4, ], [1, 2, 2, 1], 'SAME')
    up_deconv = tf.reshape(up_deconv, shape=[-1, final_size*2, final_size*2, filters // 4, ])
    up_norm = utils.Normalization(up_deconv, 'batch', cfg.training, 'final_upnorm2')
    up_act = utils.activation('final_upact2', up_norm, cfg.ACTIVATION_FUNC)

    up_conv_f = utils.conv2D('final_upconv1', up_act, 2, [1, 1], [1, 1], 'SAME')
    print('final_upconv1 : ', up_conv_f.shape)

    return up_conv_f

def residual_v2_with_unet(inputs):


    ################# Resnet v2 with unet upsample #########################

    filter_chunk = cfg.n_filter_chunks

    n_filters = []
    filters = cfg.INIT_FILTER
    for i in range(cfg.RES_DEPTH):
        if i > 0:
            filters *= 2
        else:
            pass
        n_filters.append(filters)

    print('n_filters : ', n_filters)


    conv = [0] * cfg.v2_depth
    norm = [0] * cfg.v2_depth
    act = [0] * cfg.v2_depth
    middle = [0] * cfg.v2_depth
    up_concat = [0] * cfg.v2_depth
    up_conv = [0] * cfg.v2_depth
    up_conv2 = [0] * cfg.v2_depth
    up_deconv = [0] * cfg.v2_depth
    up_norm = [0] * cfg.v2_depth
    up_act = [0] * cfg.v2_depth
    shortcut = [0] * cfg.v2_depth

    # filters = int(filter_chunk[i])
    filters = cfg.INIT_FILTER

    for n in range(len(filter_chunk)):

        iterates = cfg.n_filter_chunks[n]
        print('iterates : ', iterates)

        # if n > 0:
        #     filters *= 2

        for i in range(int(iterates)):

            # if it's first loop, don't need to shrink size because maxpool would do.
            if i == iterates - 1 :
                strides = [2,2]
            else:
                strides = [1,1]

            shortcut[i] = tf.identity(inputs, 'shortcut')
            shortcut[i] = utils.Normalization(shortcut[i], 'batch', cfg.training, str(n) + str(i) + '_short_cut_bn')
            shortcut[i] = utils.activation(str(n) + str(i) + 'sc_act1', shortcut[i], cfg.ACTIVATION_FUNC)
            shortcut[i] = utils.conv2D(str(n) + str(i) + '_short_cut', shortcut[i], filters , [1, 1], strides,
                                       padding='SAME')

            print('{2} |shortcut filter{0} : {1}'.format(str(i), filters, str(n)))

            conv[i] = utils.conv2D(str(n) + str(i) + '_conv1', inputs, filters, [3, 3], [1, 1], padding='SAME')
            norm[i] = utils.Normalization(conv[i], 'batch', cfg.training, str(n) + str(i) + '_norm1')
            act[i] = utils.activation(str(n) + str(i) + '_act1', norm[i], cfg.ACTIVATION_FUNC)
            conv[i] = utils.conv2D(str(n) + str(i) + '_conv2', act[i], filters, [3, 3], strides, padding='SAME')

            inputs = conv[i] + shortcut[i]
            print('{2} |conv filter{0} : {1}'.format(str(i), filters, str(n)))
            middle[n] = conv[int(iterates) - 1] + shortcut[int(iterates) - 1]

        filters *= 2

    # print('{3} | conv[{0}] : {1}, shortcut : {2}'.format(str(i), conv[i].shape, shortcut.shape, str(n)))

    # inputs = conv[i] + shortcut
    final_filter = filter_chunk[len(filter_chunk) - 1]

    # up_inputs = conv[len(filter_chunk)-1] + shortcut
    up_inputs = middle[len(filter_chunk) - 1]

    final_size = int(up_inputs.shape[1])
    print('final_size : ', final_size)
    # middle[n] = conv[i] + shortcut
    print('middle[{0}] : {1}'.format(str(n), middle[n].shape))

    norm = utils.Normalization(up_inputs, 'batch', cfg.training, 'norm1')
    act = utils.activation('act1', norm, cfg.ACTIVATION_FUNC)
    up_inputs= utils.conv2D('conv2', act, filters, [3, 3], [2,2], padding='SAME')


    ############## up_sample #############

    print('filters : ', filters)

    for n in reversed(range(len(filter_chunk))):
        print('filter2 : ', filters)

        up_deconv[n] = utils.deconv2D(str(n) + '_updeconv1', up_inputs, [3, 3, filters // 2, filters],
                                      [-1, final_size, final_size, filters // 2, ], [1, 2, 2, 1], 'SAME')
        up_deconv[n] = tf.reshape(up_deconv[n], shape=[-1, final_size, final_size, filters // 2, ])

        up_norm[n] = utils.Normalization(up_deconv[n], 'batch', cfg.training, str(n) + '_upnorm1')
        up_act[n] = utils.activation(str(n) + '_upact1', up_norm[n], cfg.ACTIVATION_FUNC)
        up_concat[n] = utils.concat(str(n) + '_upconcat1', [up_act[n], middle[n]], 3)

        filters //= 2
        final_size *= 2

        up_conv[n] = utils.conv2D(str(n) + '_upconv1', up_concat[n], filters, [3, 3], [1, 1], 'SAME')
        up_norm[n] = utils.Normalization(up_conv[n], 'batch', cfg.training, str(n) + '_upnorm2')
        up_act[n] = utils.activation(str(n) + '_upact1', up_norm[n], cfg.ACTIVATION_FUNC)
        up_conv2[n] = utils.conv2D(str(n) + '_upconv2', up_act[n], filters, [3, 3], [1, 1], 'SAME')
        up_norm[n] = utils.Normalization(up_conv2[n], 'batch', cfg.training, str(n) + '_upnorm3')
        up_act[n] = utils.activation(str(n) + '_upact2', up_norm[n], cfg.ACTIVATION_FUNC)
        print('{0}_updeconv1 : {1}, {0}_upconcat1 : {2}, {0}_upconv1 : {3}, {0}_upconv2 : {4}'
              .format(str(n), up_deconv[n].shape, up_concat[n].shape, up_conv[n].shape, up_conv2[n].shape))

        up_inputs = up_act[n]

    up_deconv = utils.deconv2D('final_updeconv1', up_inputs, [3, 3, filters // 2, filters],
                               [-1, final_size, final_size, filters // 2, ], [1, 2, 2, 1], 'SAME')
    up_deconv = tf.reshape(up_deconv, shape=[-1, final_size, final_size, filters // 2, ])
    up_norm = utils.Normalization(up_deconv, 'batch', cfg.training, 'final_upnorm1')
    up_act = utils.activation('final_upact1', up_norm, cfg.ACTIVATION_FUNC)
    up_deconv = utils.deconv2D('final_updeconv2', up_act, [3, 3, filters // 4, filters//2],
                                  [-1, final_size*2, final_size*2, filters // 4, ], [1, 2, 2, 1], 'SAME')
    up_deconv = tf.reshape(up_deconv, shape=[-1, final_size*2, final_size*2, filters // 4, ])
    up_norm = utils.Normalization(up_deconv, 'batch', cfg.training, 'final_upnorm2')
    up_act = utils.activation('final_upact2', up_norm, cfg.ACTIVATION_FUNC)

    up_conv_f = utils.conv2D('final_upconv1', up_act, 2, [1, 1], [1, 1], 'SAME')
    print('final_upconv1 : ', up_conv_f.shape)

    return up_conv_f
