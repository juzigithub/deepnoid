import tensorflow as tf
import config as cfg
import os
import cv2
import numpy as np


#############################################################################################################################
#                                                    Layer Functions                                                        #
#############################################################################################################################

initializer = tf.contrib.layers.variance_scaling_initializer()
regularizer = None # tf.contrib.layers.l2_regularizer(0.00001)

# ACTIVATION_FUNC = 'relu'       # relu, elu, prelu, leaky_relu

def conv2D(name, inputs, filters, kernel_size, strides, padding='valid'):
    conv2D = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                              padding=padding, use_bias=True, kernel_initializer=initializer, kernel_regularizer=regularizer, name=name)
    return conv2D


def s_conv2D(name, inputs, filters, kernel_size, strides, padding='valid'):
    s_conv2D = tf.layers.separable_conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=True,
                                          depthwise_initializer=initializer, depthwise_regularizer=regularizer, pointwise_initializer=initializer,
                                          pointwise_regularizer=regularizer, name=name)
    return s_conv2D


def deconv2D(name, inputs, filter_shape, output_shape, strides, padding='valid'):
    W = tf.get_variable(name+'W', filter_shape, initializer=initializer,regularizer=regularizer)
    shape = tf.shape(inputs)
    batch_size = shape[0]
    output_shape2 = [batch_size, output_shape[1], output_shape[2], output_shape[3]]
    layer = tf.nn.conv2d_transpose(inputs, filter=W, output_shape=output_shape2, strides=strides, padding=padding)
    return layer


def re_conv2D(name, inputs, output_shape):
    """
    https://distill.pub/2016/deconv-checkerboard/
    """
    resize_layer = tf.image.resize_nearest_neighbor(images=inputs, size=[output_shape[1], output_shape[2]], name=name+'_resizing')
    # padding_layer = tf.pad(resize_layer)
    # conv_layer = conv2D(padding_layer)
    conv_layer = conv2D(name=name+'_conv', inputs=resize_layer, filters=output_shape[3], kernel_size=[3, 3], strides=[1, 1], padding='same')
    return conv_layer


def GlobalAveragePooling2D(input, n_class, name):
    """
    replace Fully Connected Layer.
    https://www.facebook.com/groups/smartbean/permalink/1708560322490187/
    https://github.com/AndersonJo/global-average-pooling/blob/master/global-average-pooling.ipynb
    """
    gap_filter = tf.get_variable(name='gap_filter', shape=[1, 1, input.get_shape()[-1], n_class], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
    layer = tf.nn.conv2d(input, filter=gap_filter, strides=[1, 1, 1, 1], padding='SAME', name=name)
    layer = tf.nn.avg_pool(layer, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID')
    layer = tf.reduce_mean(layer, axis=[1, 2])
    return layer


def maxpool(name,inputs, pool_size, strides, padding='valid'):
    MP_layer = tf.layers.max_pooling2d(inputs, pool_size, strides, padding,name=name)
    return MP_layer


def averagepool(name,inputs, pool_size, strides, padding='valid'):
    AP_layer = tf.layers.average_pooling2d(inputs, pool_size, strides, padding,name=name)
    return AP_layer


def maxout(name,inputs, num_units):
    # num_units must multiple of axis
    MO_layer = tf.contrib.layers.maxout(inputs, num_units,name=name)
    return MO_layer


def concat(name,inputs,axis):
    con_layer = tf.concat(inputs,axis,name=name)
    return con_layer


def dropout(name,inputs,drop_rate,training):
    DP_layer = tf.layers.dropout(inputs,drop_rate,training=training,name=name)
    return DP_layer


def add(*inputs,name):
    layer = tf.add(*inputs,name=name)
    return layer


def flatten(name,inputs):
    L1 = tf.layers.flatten(inputs,name=name)
    return L1


def fully_connected(name,inputs,units):
    L2 = tf.layers.dense(inputs,units,name=name,kernel_initializer=initializer,kernel_regularizer=regularizer)
    return L2


#############################################################################################################################
#                                                 Activation Functions                                                      #
#############################################################################################################################

def activation(name, inputs, type):
    if type == 'relu':
        active_layer = tf.nn.relu(inputs,name=name)
        return active_layer

    elif type == 'lrelu':
        alpha=0.2
        x = tf.maximum(inputs, alpha * inputs, name=name)
        return x

    elif type == 'elu':
        active_layer = tf.nn.elu(inputs,name=name)
        return active_layer

    elif type == 'prelu':
        alphas = tf.get_variable(name, inputs.get_shape()[-1], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        pos = tf.nn.relu(inputs)
        neg = alphas * (inputs - abs(inputs)) * 0.5
        return pos + neg

    elif type == 'selu':
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(inputs > 0.0, inputs, alpha * tf.exp(inputs) - alpha)

    elif type is None:
        return inputs


def Normalization(x, norm_type, is_train, name, G=2, esp=1e-5, channel_mode='NHWC'):
    with tf.variable_scope('{}_norm'.format(norm_type)):
        if norm_type == 'None':
            output = x
            return output

        # Batch Normalization
        elif norm_type == 'batch':
            # output = tf.contrib.layers.batch_norm(x, center=True, scale=True, decay=0.999, is_training=is_train, updates_collections=None)
            output = tf.layers.batch_normalization(x, momentum=0.9, epsilon=0.0001, training=is_train, name='BN_'+name)
            return output

        # Group Normalization
        elif norm_type == 'group':
            with tf.name_scope('GN_'+name):
                if channel_mode == 'NHWC':
                    # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
                    x = tf.transpose(x, [0, 3, 1, 2])
                    N, C, H, W = x.get_shape().as_list()
                    G = min(G, C)

                    x = tf.reshape(x, [N, G, C // G, H, W])
                    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
                    x = (x - mean) / tf.sqrt(var + esp)
                    # per channel gamma and beta
                    gamma = tf.get_variable(name+'_GN_gamma', [C],
                                            initializer=tf.constant_initializer(1.0))
                    beta = tf.get_variable(name+'_GN_beta', [C],
                                           initializer=tf.constant_initializer(0.0))
                    gamma = tf.reshape(gamma, [1, C, 1, 1])
                    beta = tf.reshape(beta, [1, C, 1, 1])

                    output = tf.reshape(x, [N, C, H, W]) * gamma + beta
                    # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
                    output = tf.transpose(output, [0, 2, 3, 1])
                    return output

                elif channel_mode == 'NCHW':
                    N, C, H, W = x.get_shape().as_list()
                    G = min(G, C)

                    x = tf.reshape(x, [N, G, C // G, H, W])
                    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
                    x = (x - mean) / tf.sqrt(var + esp)
                    # per channel gamma and beta
                    gamma = tf.get_variable(name + '_GN_gamma', [C],
                                            initializer=tf.constant_initializer(1.0))
                    beta = tf.get_variable(name + '_GN_beta', [C],
                                           initializer=tf.constant_initializer(0.0))
                    gamma = tf.reshape(gamma, [1, C, 1, 1])
                    beta = tf.reshape(beta, [1, C, 1, 1])

                    output = tf.reshape(x, [N, C, H, W]) * gamma + beta
                    return output

        else:
            raise NotImplementedError


#############################################################################################################################
#                                                  Accuracy Functions                                                       #
#############################################################################################################################


def iou_coe(output, target, smooth=1e-5):
    # output : self.foreground_pred
    # target : self.foreground_truth
    # return : list of Batch IoU

    axis = [1, 2, 3]
    pre = tf.cast(output > 0.51, dtype=tf.float32)
    truth = tf.cast(target > 0.51, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    batch_iou = (inse + smooth) / (union + smooth)
    iou = batch_iou

    return iou, inse, pre





#############################################################################################################################
#                                                    Loss Functions                                                         #
#############################################################################################################################


def huber_loss(output, target, delta=1.0):
    residual = tf.abs(output - target)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


# def focal_loss(output, target, smooth=1e-6):
#     focal = -tf.reduce_sum(tf.square(tf.ones_like(output) - output) * target * tf.log(target + smooth))
#     return focal


def focal_loss(output, target, epsilon=1e-6):
    ndim = len(output.get_shape())
    output /= tf.reduce_sum(output, axis=(ndim - 1), keep_dims=True)
    output = tf.clip_by_value(output, epsilon, 1 - epsilon)
    focal = -tf.reduce_mean(tf.square(tf.ones_like(output)-output)*target*tf.log(output), axis=tuple(range(ndim-1)))
    return tf.reduce_sum(focal)


'''
문제 : label이 없는 경우 predict에서 픽셀을 단 하나만 집어도 로스가 매우 크게 적용된다.
대안 : inse, l, r의 reduce_sum을 reduce_mean으로 수정
1. pixel-wise로 각각 곱해준다
2. 배치단위로 각각 평균을 내준다
3. 배치별로 dice loss를 구한다
4. 배치 전체를 평균낸다

* 추가 대안
1. 틀린 픽셀의 갯수에 비례해서 로그적으로 로스가 증가하게 한다
2. 있는 걸 없다고 체크한 오답에 대해 더 큰 로스를 적용한다
'''


def dice_loss(output, target, axis=(1, 2, 3), smooth=1e-6):
    inse = tf.reduce_mean(output * target, axis=axis)
    l = tf.reduce_mean(output * output, axis=axis)
    r = tf.reduce_mean(target * target, axis=axis)
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return 1-dice


def dice_loss_sum(output, target, axis=(1, 2, 3), smooth=1e-6):
    inse = tf.reduce_sum(output * target, axis=axis)
    l = tf.reduce_sum(output * output, axis=axis)
    r = tf.reduce_sum(target * target, axis=axis)
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return 1-dice


def cross_entropy(output, target):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output))


def weighted_categorical_cross_entropy(output, target, weights, epsilon=1e-6):
    ndim = len(output.get_shape())
    ncategory = output.get_shape[-1]
    # scale predictions so class probabilities of each pixel sum to 1
    output /= tf.reduce_sum(output, axis=(ndim-1), keep_dims=True)
    output = tf.clip_by_value(output, epsilon, 1-epsilon)
    w = tf.constant(weights) * (ncategory / sum(weights))
    # first, average over all axis except classes
    cross_entropies = -tf.reduce_mean(target * tf.log(output), axis=tuple(range(ndim-1)))
    return tf.reduce_sum(w * cross_entropies)

def select_loss(mode, output, target, smooth=1e-6, weight=1, epsilon=1e-6, delta=1.0):
    if mode == 'dice':
        return dice_loss(output, target, smooth=smooth)
    elif mode == 'focal':
        return focal_loss(output, target, epsilon=epsilon)
    elif mode == 'cross_entropy':
        return cross_entropy(output, target)
    elif mode == 'dice_sum':
        return dice_loss_sum(output, target, smooth=smooth)
    elif mode == 'huber':
        return huber_loss(output, target, delta=delta)
    elif mode == 'weighted_cross_entropy':
        return weighted_categorical_cross_entropy(output, target, weights=weight)
    else:
        print("Not supported loss function. Select among dice, focal, cross_entropy, dice_sum, huber,weighted_cross_entropy")

#############################################################################################################################
#                                                      Optimizer                                                            #
#############################################################################################################################


def select_optimizer(mode, learning_rate, loss, global_step):
    if mode == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)
    elif mode == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)
    elif mode == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)
    else:
        print("Not supported optimizer. Select among adam, rmsprop, sgd")




#############################################################################################################################
#                                                    Save Functions                                                         #
#############################################################################################################################


def result_saver(path, data):
    with open(path, 'at') as f:
        f.write(data)
        f.write('\n')



##################################
#
##################################


def select_downsampling(name, down_conv, down_pool, channel_n, pool_size, mode):
    if mode == 'neighbor':
        shape = [-1, pool_size, pool_size, channel_n]
        down_pool = re_conv2D(name + '_neighbor', down_conv, shape)

    elif mode == 'maxpool':
        down_pool = maxpool(name + '_maxpool', down_conv, [2,2], [2,2], 'SAME')

    elif mode == 'avgpool':
        down_pool = averagepool(name + '_avgpool', down_conv, [2,2], [2,2], 'SAME')

    return down_pool


def select_upsampling(name, up_conv, up_pool, channel_n, pool_size, mode):
    shape = [-1, pool_size, pool_size, channel_n]

    if mode == 'resize':
      up_pool = re_conv2D(name + '_reconv', up_conv, shape)

    elif mode == 'transpose':
        up_pool = deconv2D(name + 'deconv', up_conv, [3, 3, channel_n, channel_n * 2], shape, [1,2,2,1], 'SAME')
        up_pool = tf.reshape(up_pool, shape)

    elif mode == 'add':
        up_pool1 = re_conv2D(name + '_reconv', up_conv, shape)
        up_pool2 = deconv2D(name + 'deconv', up_conv, [3, 3, channel_n, channel_n * 2], shape, [1,2,2,1], 'SAME')
        up_pool2 = tf.reshape(up_pool2, shape)
        up_pool = up_pool1 + up_pool2

    elif mode == 'concat':
        up_pool1 = re_conv2D(name + '_reconv', up_conv, shape)
        up_pool2 = deconv2D(name + 'deconv', up_conv, [3, 3, channel_n, channel_n * 2], shape, [1, 2, 2, 1], 'SAME')
        up_pool2 = tf.reshape(up_pool2, shape)
        up_pool = concat(name + '_upsampling_concat', [up_pool1, up_pool2], axis=3)
        up_pool = conv2D(name + '_bottleneck', up_pool, channel_n, [1,1], [1,1], padding='SAME')

    return up_pool


def unet_down_block(inputs, conv_list, pool_list, channel_n, pool_size, group_n, training, idx):
    conv_list[idx] = conv2D(str(idx) + '_downconv1', inputs, channel_n, [3, 3], [1, 1], padding='SAME')
    conv_list[idx] = Normalization(conv_list[idx], cfg.NORMALIZATION_TYPE, training, str(idx) + '_downnorm1', G=group_n)
    conv_list[idx] = activation(str(idx) + '_downact1', conv_list[idx], cfg.ACTIVATION_FUNC)

    conv_list[idx] = conv2D(str(idx) + '_downconv2', conv_list[idx], channel_n, [3, 3], [1, 1], padding='SAME')
    conv_list[idx] = Normalization(conv_list[idx], cfg.NORMALIZATION_TYPE, training, str(idx) + '_downnorm2', G=group_n)
    conv_list[idx] = activation(str(idx) + '_downact2', conv_list[idx], cfg.ACTIVATION_FUNC)
    print('down' + str(idx + 1) + 'conv', conv_list[idx])
    pool_list[idx] = select_downsampling(str(idx) + '_downsampling',
                                               conv_list[idx],
                                               pool_list[idx],
                                               channel_n,
                                               pool_size,
                                               cfg.DOWNSAMPLING_TYPE)
    print('down' + str(idx + 1) + 'pool', pool_list[idx])

    if cfg.DOWNSAMPLING_TYPE == 'neighbor':
        conv_list[idx] = Normalization(conv_list[idx], cfg.NORMALIZATION_TYPE, training,
                                             str(idx) + '_norm3', G=group_n)
        conv_list[idx] = activation(str(idx) + '_act3', conv_list[idx], cfg.ACTIVATION_FUNC)

    return pool_list[idx]


def unet_same_block(inputs, channel_n, group_n, training):
    conv_list = conv2D('same_conv1', inputs, channel_n, [3, 3], [1, 1], padding='SAME')
    conv_list = Normalization(conv_list, cfg.NORMALIZATION_TYPE, training, 'same_norm1', G=group_n)
    conv_list = activation('same_act1', conv_list, cfg.ACTIVATION_FUNC)
    conv_list = conv2D('same_conv2', conv_list, channel_n, [1, 1], [1, 1], padding='SAME')
    conv_list = Normalization(conv_list, cfg.NORMALIZATION_TYPE, training, 'same_norm2', G=group_n)
    conv_list = activation('same_act2', conv_list, cfg.ACTIVATION_FUNC)

    return conv_list


def unet_up_block(inputs, downconv_list, upconv_list, pool_list, channel_n, pool_size, group_n, training, idx):
    pool_list[idx] = select_upsampling(str(idx) + '_upsampling', inputs, pool_list[idx], channel_n, pool_size,
                                             cfg.UPSAMPLING_TYPE)
    pool_list[idx] = Normalization(pool_list[idx], cfg.NORMALIZATION_TYPE, training, str(idx) + '_norm1',
                                         G=group_n)
    pool_list[idx] = activation(str(idx) + '_upsampling_act', pool_list[idx], cfg.ACTIVATION_FUNC)
    print('up' + str(idx + 1) + 'pool', pool_list[idx])

    pool_list[idx] = concat(str(idx) + '_upconcat', [pool_list[idx], downconv_list[idx]], axis=3)
    print('up' + str(idx + 1) + 'concat', pool_list[idx])

    upconv_list[idx] = conv2D(str(idx) + '_upconv1', pool_list[idx], channel_n, [3, 3], [1, 1], padding='SAME')
    upconv_list[idx] = Normalization(upconv_list[idx], cfg.NORMALIZATION_TYPE, training, str(idx) + '_upnorm1',
                                           G=group_n)
    upconv_list[idx] = activation(str(idx) + '_upact1', upconv_list[idx], cfg.ACTIVATION_FUNC)
    upconv_list[idx] = conv2D(str(idx) + '_upconv2', upconv_list[idx], channel_n, [3, 3], [1, 1], padding='SAME')
    upconv_list[idx] = Normalization(upconv_list[idx], cfg.NORMALIZATION_TYPE, training, str(idx) + '_upnorm2',
                                           G=group_n)
    upconv_list[idx] = activation(str(idx) + '_upact2', upconv_list[idx], cfg.ACTIVATION_FUNC)
    print('up' + str(idx + 1) + 'conv', upconv_list[idx])

    return upconv_list[idx]

def residual_block_v1(inputs, channel_n, group_n, training, idx, shortcut=True):
    # bottleneck1
    hl = conv2D(str(idx) + '_bottleneck1', inputs, int(channel_n/4), [1, 1], [1, 1], padding='SAME')
    hl = Normalization(hl, cfg.NORMALIZATION_TYPE, training, str(idx) + '_bottleneck_norm1', G=group_n)
    hl = activation(str(idx) + '_bottleneck_act1', hl, cfg.ACTIVATION_FUNC)

    # conv
    hl = conv2D(str(idx) + '_conv', hl, int(channel_n / 4), [3, 3], [1, 1], padding='SAME')
    hl = Normalization(hl, cfg.NORMALIZATION_TYPE, training, str(idx) + '_conv_norm', G=group_n)
    hl = activation(str(idx) + '_conv_act', hl, cfg.ACTIVATION_FUNC)

    # bottleneck2
    hl = conv2D(str(idx) + '_bottleneck2', inputs, channel_n, [1, 1], [1, 1], padding='SAME')
    hl = Normalization(hl, cfg.NORMALIZATION_TYPE, training, str(idx) + '_bottleneck_norm2', G=group_n)
    hl = activation(str(idx) + '_bottleneck_act2', hl, cfg.ACTIVATION_FUNC)

    hl = inputs + hl if shortcut else hl

    return hl


def dense_layer(name, inputs, group_n, drop_rate, training, idx):
    # bottleneck
    l = Normalization(inputs, cfg.NORMALIZATION_TYPE, training, name + str(idx) + '_bottleneck_norm1', G=group_n)
    l = activation(name + str(idx) + '_bottleneck_act1', l, cfg.ACTIVATION_FUNC)
    l = conv2D(name + str(idx) + '_bottleneck1', l, 4 * cfg.GROWTH_RATE, [1, 1], [1, 1], padding='SAME')
    l = dropout(name + str(idx) + '_dropout1', l, drop_rate, training)

    # conv
    l = Normalization(l, cfg.NORMALIZATION_TYPE, training, name + str(idx) + '_bottleneck_norm2', G=group_n)
    l = activation(name + str(idx) + '_bottleneck_act2', l, cfg.ACTIVATION_FUNC)
    l = conv2D(name + str(idx) + '_bottleneck2', l, cfg.GROWTH_RATE, [3, 3], [1, 1], padding='SAME')
    l = dropout(name + str(idx) + '_dropout2', l, drop_rate, training)

    return l

def transition_layer(name, inputs, group_n, training, specific_n_channels=False, idx=0):
    if specific_n_channels:
        l = Normalization(inputs, cfg.NORMALIZATION_TYPE, training, name + str(idx) + '_bottleneck_norm1', G=group_n)
        l = activation(name + str(idx) + '_bottleneck_act1', l, cfg.ACTIVATION_FUNC)
        l = conv2D(name + str(idx) + '_bottleneck1', l, specific_n_channels[idx], [1, 1], [1, 1], padding='SAME')

    else:
        n_channels = inputs.get_shape().as_list()[-1]

        l = Normalization(inputs, cfg.NORMALIZATION_TYPE, training, name + str(idx) + '_bottleneck_norm1', G=group_n)
        l = activation(name + str(idx) + '_bottleneck_act1', l, cfg.ACTIVATION_FUNC)
        l = conv2D(name + str(idx) + '_bottleneck1', l, n_channels * cfg.THETA, [1, 1], [1, 1], padding='SAME')

    return l

def dense_block_v1(name, inputs, group_n, drop_rate, training, n_layer):
    hl = tf.identity(inputs)

    for idx in range(n_layer):
        l = dense_layer(name, hl, group_n, drop_rate, training, idx)
        hl = tf.concat([hl, l], axis=3)

    return hl

def mobilenet():
    pass

def shufflenet():
    pass

def henet():
    pass

def depthwise_separable_convlayer(name, inputs, channel_n, width_mul, group_n, training, idx):
    # depthwise
    depthwise_filter = tf.get_variable(name='depthwise_filter', shape=[1, 1, inputs.get_shape()[-1], width_mul],
                                       dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
    l = tf.nn.depthwise_conv2d(inputs, depthwise_filter, strides = [1, 3, 3, 1], padding = 'SAME', name = name + str(idx) + '_depthwise')
    l = Normalization(l, cfg.NORMALIZATION_TYPE, training, name + str(idx) + '_depthwise_norm', G=group_n)
    l = activation(name + str(idx) + '_depthwise_act1', l, cfg.ACTIVATION_FUNC)

    # pointwise
    l = conv2D(name + str(idx) + '_pointwise', l, channel_n, [1, 1], [1, 1], padding='SAME')
    l = Normalization(l, cfg.NORMALIZATION_TYPE, training, name + str(idx) + '_pointwise_norm1', G=group_n)
    l = activation(name + str(idx) + '_pointwise_act1', l, cfg.ACTIVATION_FUNC)

    return l

