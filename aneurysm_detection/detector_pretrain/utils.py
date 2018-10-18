# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff
import nibabel
from tensorflow.python.ops import array_ops
from skimage.feature import greycomatrix, greycoprops
from sklearn.feature_extraction import image
import os
import SimpleITK as sitk
from itertools import product
from skimage.exposure import rescale_intensity
from scipy.signal import argrelextrema
import math

#############################################################################################################################
#                                                    Layer Functions                                                        #
#############################################################################################################################

initializer = tf.contrib.layers.variance_scaling_initializer()
regularizer = tf.contrib.layers.l2_regularizer(0.0001)


def conv2D(name, inputs, filters, kernel_size, strides, padding='valid', dilation_rate=(1,1)):
    conv2D = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                              padding=padding, use_bias=True, dilation_rate=dilation_rate,
                              kernel_initializer=initializer, kernel_regularizer=regularizer, name=name)
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

def GlobalAveragePooling2D(input, n_class, name, keep_dims=False):
    """
    replace Fully Connected Layer.
    https://www.facebook.com/groups/smartbean/permalink/1708560322490187/
    https://github.com/AndersonJo/global-average-pooling/blob/master/global-average-pooling.ipynb
    """
    kernel_size = input.get_shape().as_list()[1]
    gap_filter = tf.get_variable(name='gap_filter', shape=[1, 1, input.get_shape().as_list()[-1], n_class], dtype=tf.float32, initializer=initializer)
    layer = tf.nn.conv2d(input, filter=gap_filter, strides=[1, 1, 1, 1], padding='SAME', name=name)
    layer = tf.nn.avg_pool(layer, ksize=[1, kernel_size, kernel_size, 1], strides=[1, 1, 1, 1], padding='VALID')
    if not keep_dims:
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


def Normalization(x, norm_type, is_train, name, G=2, esp=1e-5, channel_mode='NHWC', batch_size = 10, threshold = 'fuzzy',n_divide = 4,standard=False, scale=1,first=True):
    with tf.variable_scope('{}_norm'.format(norm_type)):

        if norm_type == 'None':
            output = x
            return output

        # Batch Normalization
        elif norm_type == 'batch':
            output = tf.contrib.layers.batch_norm(inputs=x,
                                                  center=True,
                                                  scale=True,
                                                  fused=True,
                                                  decay=0.9,
                                                  is_training=is_train,
                                                  updates_collections=None,
                                                  zero_debias_moving_mean=True,
                                                  scope='BN_'+name)
            # output = tf.contrib.layers.batch_norm(inputs=x,
            #                                       center=True,
            #                                       scale=True,
            #                                       fused=True,
            #                                       decay=0.001,
            #                                       is_training=is_train,
            #                                       updates_collections=None,
            #                                       zero_debias_moving_mean=False,
            #                                       scope='BN_'+name)
            # output = tf.layers.batch_normalization(x, momentum=0.9, epsilon=0.0001, training=is_train, name='BN_'+name)
            return output

        # Group Normalization
        elif norm_type == 'group':
            with tf.name_scope('GN_'+name):
                if channel_mode == 'NHWC':

                    # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
                    x = tf.transpose(x, [0, 3, 1, 2])
                    N, C, H, W = x.get_shape().as_list()
                    G = min(G, C)
                    # N -> -1
                    x = tf.reshape(x, [-1, G, C // G, H, W])
                    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
                    x = (x - mean) / tf.sqrt(var + esp)
                    # per channel gamma and beta
                    gamma = tf.get_variable(name+'_GN_gamma', [C],
                                            initializer=tf.constant_initializer(1.0))
                    beta = tf.get_variable(name+'_GN_beta', [C],
                                           initializer=tf.constant_initializer(0.0))
                    gamma = tf.reshape(gamma, [1, C, 1, 1])
                    beta = tf.reshape(beta, [1, C, 1, 1])
                    # N -> -1
                    output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
                    # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
                    output = tf.transpose(output, [0, 2, 3, 1])
                    return output

                elif channel_mode == 'NCHW':
                    N, C, H, W = x.get_shape().as_list()
                    G = min(G, C)
                    # N -> -1
                    x = tf.reshape(x, [-1, G, C // G, H, W])
                    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
                    x = (x - mean) / tf.sqrt(var + esp)
                    # per channel gamma and beta
                    gamma = tf.get_variable(name + '_GN_gamma', [C],
                                            initializer=tf.constant_initializer(1.0))
                    beta = tf.get_variable(name + '_GN_beta', [C],
                                           initializer=tf.constant_initializer(0.0))
                    gamma = tf.reshape(gamma, [1, C, 1, 1])
                    beta = tf.reshape(beta, [1, C, 1, 1])
                    # N -> -1
                    output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
                    return output
        elif norm_type == 'batch_match':
            output = batch_histogram_match_tensor(x, batch_size=batch_size, threshold = threshold, n_divide = n_divide, standard=standard, scale=scale,first=first)
            return output

        else:
            raise NotImplementedError


#############################################################################################################################
#                                                  Accuracy Functions                                                       #
#############################################################################################################################

def cal_result_detection(class_pred, bbox_pred, class_label, bbox_label):
    match_list = []
    iou_list = []
    for pc, pb in zip(class_pred, bbox_pred):
        iou = 0
        result = [0, 0]

        if np.sum(pb) > 0:
            for gc, gb in zip(class_label, bbox_label):
                if int(pc) == int(gc):
                    ins_y1 = np.maximum(pb[0], gb[0])
                    ins_x1 = np.maximum(pb[1], gb[1])
                    ins_y2 = np.minimum(pb[2], gb[2])
                    ins_x2 = np.minimum(pb[3], gb[3])

                    ins_h = np.maximum(ins_y2 - ins_y1, 0)
                    ins_w = np.maximum(ins_x2 - ins_x1, 0)
                    ins = ins_h * ins_w

                    p_h = pb[2] - pb[0]
                    p_w = pb[3] - pb[1]
                    pred = p_h * p_w

                    g_h = gb[2] - gb[0]
                    g_w = gb[3] - gb[1]
                    gt = g_h * g_w

                    result = [1, (ins + 1e-6) / (pred + gt - ins + 1e-6)] if ((ins + 1e-6) / (pred + gt - ins + 1e-6)) > iou else result
                    iou = result[1]

            match_list.append(result[0])
            iou_list.append(result[1])
            print('utils.cal_result_detection', match_list)
            print('utils.cal_result_detection', iou_list)

    mean_match = np.mean(np.array(match_list))
    mean_iou = np.mean(np.array(iou_list))

    return [mean_match, mean_iou]

#############################################################################################################################
#                                                    Loss Functions                                                         #
#############################################################################################################################

def cross_entropy(output, target):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output))

#############################################################################################################################
#                                                      Optimizer                                                            #
#############################################################################################################################


def select_optimizer(mode, learning_rate, loss, global_step, var_list=None):
    if mode == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, var_list=var_list, global_step=global_step)
    elif mode == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss=loss, var_list=var_list, global_step=global_step)
    elif mode == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss, var_list=var_list, global_step=global_step)
    else:
        print("Not supported optimizer. Select among adam, rmsprop, sgd")


#############################################################################################################################
#                                                  Sampling Functions                                                       #
#############################################################################################################################

def select_upsampling(name, up_conv, up_pool, channel_n, pool_size_h, pool_size_w, mode):
    shape = [-1, pool_size_h, pool_size_w, channel_n]

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



#############################################################################################################################
#                                                    Save Functions                                                         #
#############################################################################################################################

def result_saver(path, data):
    with open(path, 'at') as f:
        f.write(data)
        f.write('\n')

def save_array_as_nifty_volume(data, filename):
   img = nibabel.Nifti1Image(data, affine=np.eye(4))
   nibabel.save(img, filename)

#############################################################################################################################
#                                                      CNN Models                                                           #
#############################################################################################################################

def residual_block_dw_dr(name, inputs, channel_n, width_mul, group_n, drop_rate, act_fn, norm_type, training, idx, rate=None, shortcut=True):
    # input
    if shortcut:
        il = conv2D(name + str(idx) + '_input', inputs, channel_n, [1, 1], [1, 1], padding='SAME')
        il = Normalization(il, norm_type, training, name + str(idx) + '_input_norm', G=group_n)
        il = activation(name + str(idx) + '_input_act', il, act_fn)
    else:
        il = inputs

    # bottleneck1
    hl = conv2D(name + str(idx) + '_bottleneck1', inputs, int(channel_n/4), [1, 1], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, name + str(idx) + '_bottleneck_norm1', G=group_n)
    hl = activation(name + str(idx) + '_bottleneck_act1', hl, act_fn)
    hl = dropout(name + str(idx) + '_dropout1', hl, drop_rate, training)

    # depthwise
    depthwise_filter = tf.get_variable(name=name+'depthwise_filter' + str(idx),
                                       shape=[3, 3, int(channel_n/4), width_mul],
                                       dtype=tf.float32,
                                       initializer=initializer)  # [filter_height, filter_width, in_channels, channel_multiplier]
    hl = tf.nn.depthwise_conv2d(hl, depthwise_filter, [1, 1, 1, 1], 'SAME', rate=rate, name = name + str(idx) + '_depthwise')
    hl = Normalization(hl, norm_type, training, name + str(idx) + '_depthwise_norm', G=group_n)
    hl = activation(name + str(idx) + '_depthwise_act', hl, act_fn)
    hl = dropout(name + str(idx) + '_dropout2', hl, drop_rate, training)

    # bottleneck2
    hl = conv2D(name + str(idx) + '_bottleneck2', hl, channel_n, [1, 1], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, name + str(idx) + '_bottleneck_norm2', G=group_n)
    hl = activation(name + str(idx) + '_bottleneck_act2', hl, act_fn)
    hl = dropout(name + str(idx) + '_dropout3', hl, drop_rate, training)

    hl = il + hl

    return hl

def depthwise_separable_convlayer_dr(name, inputs, channel_n, width_mul, group_n, drop_rate, act_fn, norm_type, training, idx,  rate=None):
    # depthwise
    depthwise_filter = tf.get_variable(name=name + 'depthwise_filter' + str(idx),
                                       shape=[3, 3, inputs.get_shape()[-1], width_mul],
                                       dtype=tf.float32,
                                       initializer=initializer)
    l = tf.nn.depthwise_conv2d(inputs, depthwise_filter, [1, 1, 1, 1], 'SAME', rate=rate,
                               name=name + str(idx) + '_depthwise')
    l = Normalization(l, norm_type, training, name + str(idx) + '_depthwise_norm', G=group_n)
    l = activation(name + str(idx) + '_depthwise_act1', l, act_fn)
    l = dropout(name + str(idx) + '_dropout2', l, drop_rate, training)

    # pointwise
    l = conv2D(name + str(idx) + '_pointwise', l, channel_n, [1, 1], [1, 1], padding='SAME')
    l = Normalization(l, norm_type, training, name + str(idx) + '_pointwise_norm1', G=group_n)
    l = activation(name + str(idx) + '_pointwise_act1', l, act_fn)
    l = dropout(name + str(idx) + '_dropout2', l, drop_rate, training)

    return l

#############################################################################################################################
#                                                    Result Function                                                        #
#############################################################################################################################

def masking_rgb(img, color=None, multiply=255):
    if len(np.shape(img)) <= 2:
        _img = np.expand_dims(img, axis=3)
    else:
        _img = img
    rgb_list = [np.zeros(np.shape(_img)) for _ in range(3)]

    if color == 'yellow':
        rgb_list[1] = _img
        rgb_list[2] = _img
        B, G, R = rgb_list

    elif color != None:
        rgb_dic = {'blue': 0, 'green': 1, 'red': 2}
        rgb_list[rgb_dic[color]] = _img
        B, G, R = rgb_list
    else:
        B = G = R = _img

    concat_img = np.concatenate((B, G, R), axis=-1)
    out_img = concat_img * multiply

    return out_img

#############################################################################################################################
#                                        Feature Extraction Function                                                        #
#############################################################################################################################

def save_array_as_nifty_volume2(data, filename):
    '''
    :param data: np array
    :param filename: path + filename want to save
    '''
    img = nibabel.Nifti1Image(data, affine=np.eye(4))
    nibabel.save(img, filename)

def get_array_as_nifty_volume(data):
    nii = nibabel.Nifti1Image(data, affine=np.eye(4))
    return nii

#############################################################################################################################
#                                              Histogram Function                                                           #
#############################################################################################################################

def cal_hm_landmark(arr, max_percent = 99.8, threshold = 'fuzzy_log', n_divide = 4, standard=False, scale=1):
    if arr.ndim > 1:
         arr = arr.ravel()
    arr_hist_sd, arr_edges_sd = np.histogram(arr, bins = range(np.max(arr)+2))

    hist_mean = int(np.mean(arr))
    black_peak = np.argmax(arr_hist_sd[:hist_mean])
    white_peak = hist_mean + np.argmax(arr_hist_sd[hist_mean:])

    ## Valley : Error Occured
    # valley = arr_hist_sd[int(black_peak):int(white_peak)]
    # # consider only points over 10
    # over_cutline = np.where(valley > 10)
    # # find local minmums among 500 points
    # local_mins = argrelextrema(valley[over_cutline], np.less, order=250)
    # # take first local minimum
    # local_min = over_cutline[0][local_mins[0][0]]
    # # local_min = argrelextrema(valley, np.less, order=250)[0][np.where(arr_hist_sd[argrelextrema(valley, np.less, order=250)] > 10)[0][0]]

    # define coordinate of black_peak = (a, b) and white_peak = (c, d)
    # fuzzy_log = nodal point of [(a, log(d)), (c, 0)] and [(c, log(b)), (a, 0)]
    black_peak_val = np.log(arr_hist_sd[black_peak])
    white_peak_val = np.log(arr_hist_sd[white_peak])
    fuzzy_log = ((black_peak * black_peak_val) + (white_peak * white_peak_val)) / (black_peak_val + white_peak_val)

    threshold_dict = {}      # 'fuzzy', 'fuzzy_log', 'mean', 'median', 'valley'
    threshold_dict['fuzzy'] = int((black_peak + white_peak) / 2)
    threshold_dict['fuzzy_log'] = fuzzy_log
    threshold_dict['mean'] = hist_mean
    threshold_dict['median'] = int(np.median(arr))
    # threshold_dict['valley'] = black_peak + local_min

    pc1 = threshold_dict[threshold]
    pc2 = np.percentile(arr, max_percent)
    if pc1 > pc2:
        print('pc1({0}) > pc2({1}'.format(pc1, pc2))
        pc1 = np.percentile(arr, max_percent - 20)
    ioi = arr[np.where((arr>=pc1) * (arr<=pc2))]
    landmark_list = [np.percentile(ioi, i * (100/n_divide) ) for i in range(n_divide) if not i == 0]
    landmark_list = [pc1] + landmark_list + [pc2]

    if standard:
         std_scale = (scale / pc2)
         landmark_list = [landmark * std_scale for landmark in landmark_list]

    return [int(landmark) for landmark in landmark_list]


def hm_rescale(arr, input_landmark_list, standard_landmark_list):
    arr_shape = arr.shape
    if arr.ndim > 1:
        arr = arr.ravel()
    arr_copy = np.zeros_like(arr)

    scale_idx = np.where((arr < input_landmark_list[0]))

    # 0 ~ pc1 rescale
    arr_copy[scale_idx] = rescale_intensity(arr[scale_idx],
                                            in_range=(input_landmark_list[0] - 1, input_landmark_list[0]),
                                            out_range=(standard_landmark_list[0] - 1, standard_landmark_list[0]))
    # pc1 ~ m25 ~ m50 ~ m75 ~ pc2 rescale
    for idx in range(len(input_landmark_list) - 1):

        scale_idx = np.where((arr >= input_landmark_list[idx]) * (arr < input_landmark_list[idx+1]))
        arr_copy[scale_idx] = rescale_intensity(arr[scale_idx],
                                                in_range=(input_landmark_list[idx], input_landmark_list[idx+1]),
                                                out_range=(standard_landmark_list[idx], standard_landmark_list[idx+1]))
    # pc2 ~ max rescale
    scale_idx = np.where((arr >= input_landmark_list[-1]))
    arr_copy[scale_idx] = rescale_intensity(arr[scale_idx],
                                                in_range=(input_landmark_list[-1], input_landmark_list[-1] + 1),
                                                out_range=(standard_landmark_list[-1], standard_landmark_list[-1] + 1))



    arr_copy = np.clip(arr_copy, a_min=standard_landmark_list[0], a_max=standard_landmark_list[-1])

    return arr_copy.reshape(arr_shape)

#############################################################################################################################
#                                        Regional Proposal Layer                                                            #
#############################################################################################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result

def apply_box_deltas_graph2(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """

    # (gt_center_y - a_center_y) / config.IMG_SIZE[0] * 100,
    # (gt_center_x - a_center_x) / config.IMG_SIZE[0] * 100,
    # (gt_h - a_h) / config.IMG_SIZE[0] * 100,
    # (gt_w - a_w) / config.IMG_SIZE[0] * 100,
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] / 100
    center_x += deltas[:, 1] / 100
    height += deltas[:, 2] / 100 ##################### deltas[:, 2] / 100
    width += deltas[:, 3] / 100 ##################### deltas[:, 3] / 100
    # Convert back to y1, x1, y2, x2
    # y1 = center_y - 0.5 * height
    # x1 = center_x - 0.5 * width
    # y2 = y1 + height
    # x2 = x1 + width
    y1 = center_y - 2 * 0.5 * height
    x1 = center_x - 2 * 0.5 * width
    y2 = center_y + 2 * 0.5 * height
    x2 = center_x + 2 * 0.5 * width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result

def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes (넓이)
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union

    return iou

def get_anchors(image_shape, scales, ratios, feature_strides, anchor_stride, normalization=True):
    """Returns anchor pyramid for the given image size."""
    feature_shapes = compute_backbone_shapes(image_shape, feature_strides)
    a = generate_pyramid_anchors(
        scales,
        ratios,
        feature_shapes,
        feature_strides,
        anchor_stride)
    # Normalize coordinates
    if normalization:
        a = norm_boxes(a, image_shape)
    return a

def compute_backbone_shapes(image_shape, backbone_strides):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    # backbone_stride = [4, 8, 16, 32, 64]
    return np.array([[int(math.ceil(image_shape[0] / stride)), int(math.ceil(image_shape[1] / stride))]
                     for stride in backbone_strides])

def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[0],
                                        feature_strides[0], anchor_stride))

    # for i in range(len(scales)):
    #     anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
    #                                     feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    # scales : (1,) -> (3,) [128, 128, 128]
    # ratios : (3,) -> (3,) [0.5, 1, 2]
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    # shape : feature_shape
    # feature_stride : feature_stride
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)

    return boxes

def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)

def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(tf.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Crossentropy loss
    # input shapeerror 뜨면 target/logits 부분 사용하기
    # targets = tf.cast(tf.flatten(anchor_class), 'int64')
    # logits = tf.reshape(rpn_class_logits, [-1, int(rpn_class_logits.shape[-1])])

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=anchor_class,
                                                          logits=rpn_class_logits,
                                                          )
    # loss = K.sparse_categorical_crossentropy(target=anchor_class,
    #                                          output=rpn_class_logits,
    #                                          from_logits=True)
    loss = tf.cond(tf.size(loss) > 0,
                   lambda: tf.reduce_mean(loss),
                   lambda: tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    config: the model config object.
    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = tf.squeeze(rpn_match, -1)
    indices = tf.where(tf.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = tf.reduce_sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    # TODO: use smooth_l1_loss() rather than reimplementing here
    #       to reduce code duplication
    diff = tf.abs(target_bbox - rpn_bbox)

    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)

    loss = tf.cond(tf.size(loss) > 0,
                   lambda: tf.reduce_mean(loss),
                   lambda: tf.constant(0.0))
    return loss


def detector_class_loss_graph(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """
    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Computer loss mean.
    loss = tf.reduce_sum(loss)
    return loss


def detector_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox, config):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """

    def smooth_l1_loss(y_true, y_pred):
        """Implements Smooth-L1 loss.
        y_true and y_pred are typicallly: [N, 4], but could be any shape.
        """
        diff = tf.abs(y_true - y_pred)
        less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
        loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
        return loss

    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))

    pred_bbox = tf.reshape(pred_bbox, (-1, config.TRAIN_ROIS_PER_IMAGE * config.N_CLASS, 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indicies.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = tf.cond(tf.size(target_bbox) > 0,
                   lambda: smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                   lambda: tf.constant(0.0))
    loss = tf.reduce_mean(loss)

    return loss


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

def build_rpn_targets(anchors, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    오버랩 계산/positive anchors 계산
    rpn_match : anchor들과 gt_box를 비교하여 positive/negative 판별
    rpn_bbox : delta : anchor box를 gt_box에 맞추기 위한 보정치(delta) 계산

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[anchor_iou_max < 0.3] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]
        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = (a[2] - a[0])
        a_w = (a[3] - a[1])
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    return rpn_match, rpn_bbox

def build_rpn_targets2(anchors, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    오버랩 계산/positive anchors 계산
    rpn_match : anchor들과 gt_box를 비교하여 positive/negative 판별
    rpn_bbox : delta : anchor box를 gt_box에 맞추기 위한 보정치(delta) 계산

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[anchor_iou_max < 0.3] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]
        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = (a[2] - a[0])
        a_w = (a[3] - a[1])
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / config.IMG_SIZE[0] * 100,
            (gt_center_x - a_center_x) / config.IMG_SIZE[0] * 100,
            (gt_h - a_h) / config.IMG_SIZE[0] * 100,
            (gt_w - a_w) / config.IMG_SIZE[0] * 100,
        ]
        # Normalize
        # rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    # print(rpn_bbox)
    return rpn_match, rpn_bbox

def roi_pooling(roi_proposals, conv_feature_map, pooled_shape, feature_pyramid=False):
    '''
    :param roi_proposals: [batch, num_boxes, (y1, x1, y2, x2)]
    :param conv_feature_map: [p2, p3, p4, p5 ..]
    :param pooled_shape: [pooled_width, pooled_height]
    :param feature_pyramid: conv_feature_map is feature_pyramid -> True
    :return: [batch, num_boxes, height, width, channels]
    '''

    def log2_graph(x):
        return tf.log(x) / tf.log(2.0)

    if feature_pyramid:
        n_feature = len(conv_feature_map)

        y1, x1, y2, x2 = tf.split(roi_proposals, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        # image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        # roi_level = 4 + tf.cast(tf.round(log2_graph(tf.sqrt(h * w))), tf.int32)
        # # 2 <= roi_level <= 5
        # roi_level = tf.minimum(5, tf.maximum(2, roi_level))
        # roi_level = tf.squeeze(roi_level, 2)

        roi_level = n_feature - 2 + tf.cast(tf.round(log2_graph(tf.sqrt(h * w))), tf.int32)
        # 0 <= roi_level <= n_feature-1
        roi_level = tf.minimum(n_feature - 1, tf.maximum(0, roi_level))
        roi_level = tf.squeeze(roi_level, 2)


        pooled = []
        box_to_level = []

        # for i, level in enumerate(range(2, 6)):
        #     ix = tf.where(tf.equal(roi_level, level))
        #     level_boxes = tf.gather_nd(roi_proposals, ix)
        #
        #     # Box indicies for crop_and_resize.
        #     box_indices = tf.cast(ix[:, 0], tf.int32)
        #
        #     # Keep track of which box is mapped to which level
        #     box_to_level.append(ix)
        #
        #     # Stop gradient propogation to ROI proposals
        #     level_boxes = tf.stop_gradient(level_boxes)
        #     box_indices = tf.stop_gradient(box_indices)
        #
        #     # Crop and Resize
        #     # From Mask R-CNN paper: "We sample four regular locations, so
        #     # that we can evaluate either max or average pooling. In fact,
        #     # interpolating only a single value at each bin center (without
        #     # pooling) is nearly as effective."
        #     # Result: [batch * num_boxes, pool_height, pool_width, channels]
        #     pooled.append(tf.image.crop_and_resize(conv_feature_map[i], level_boxes, box_indices, pooled_shape, method="bilinear"))
        for i in range(n_feature):
            ix = tf.where(tf.equal(roi_level, i))
            level_boxes = tf.gather_nd(roi_proposals, ix)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(conv_feature_map[i], level_boxes, box_indices, pooled_shape, method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)

    else :
        box_indices = tf.zeros((tf.shape(roi_proposals)[0] * tf.shape(roi_proposals)[1],), dtype=tf.int32)
        roi_proposals = tf.squeeze(roi_proposals, axis=0)
        pooled = tf.image.crop_and_resize(conv_feature_map, roi_proposals, box_indices, pooled_shape, method="bilinear")

    return pooled


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result

def box_refinement_graph2(box, gt_box, config):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / config.IMG_SIZE[0] * 100
    dx = (gt_center_x - center_x) / config.IMG_SIZE[0] * 100
    dh = (gt_height - height) / config.IMG_SIZE[0] * 100
    dw = (gt_width - width) / config.IMG_SIZE[0] * 100

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result

# def detection_targets_graph(proposals, gt_class_ids, gt_boxes, config):
#     """Generates detection targets for one image. Subsamples proposals and
#     generates target class IDs, bounding box deltas, and masks for each.
#
#     Inputs:
#     proposals: [Batch_size, N, (y1, x1, y2, x2)] in normalized coordinates. Might
#                be zero padded if there are not enough proposals.
#     gt_class_ids: [MAX_GT_INSTANCES] int class IDs
#     gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
#
#     Returns: Target ROIs and corresponding class IDs, bounding box shifts,
#     and masks.
#     rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
#     class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
#     deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
#             Class-specific bbox refinements.
#
#     Note: Returned arrays might be zero padded if not enough target ROIs.
#     """
#     # Assertions
#     asserts = [
#         tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
#                   name="roi_assertion"),
#     ]
#     with tf.control_dependencies(asserts):
#         proposals = tf.squeeze(proposals, axis=0) ############################################
#         proposals = tf.identity(proposals)
#
#     # Remove zero padding
#     proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
#     gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
#     gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
#                                    name="trim_gt_class_ids")
#
#     # # Handle COCO crowds
#     # # A crowd box in COCO is a bounding box around several instances. Exclude
#     # # them from training. A crowd box is given a negative class ID.
#     # crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
#     # non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
#     # crowd_boxes = tf.gather(gt_boxes, crowd_ix)
#     # gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
#     # gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
#
#     # Compute overlaps matrix [proposals, gt_boxes]
#     #############################################################
#     # overlaps = overlaps_graph(proposals, gt_boxes)
#     #
#     overlaps = overlaps_graph(tf.round(proposals * config.IMG_SIZE[0]),
#                               tf.round(gt_boxes * config.IMG_SIZE[0]))
#
#
#     # Compute overlaps with crowd boxes [anchors, crowds]
#     # crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
#     # crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
#     # no_crowd_bool = (crowd_iou_max < 0.001)
#
#     # Determine postive and negative ROIs
#     roi_iou_max = tf.reduce_max(overlaps, axis=1)
#     # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
#     positive_roi_bool = (roi_iou_max >= 0.1) ################################### 0.5
#     positive_indices = tf.where(positive_roi_bool)[:, 0] #####################################################
#     # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
#     negative_indices = tf.where(roi_iou_max < 0.1)[:, 0] ################################## 0.5
#
#     # Subsample ROIs. Aim for 33% positive
#     # Positive ROIs
#     positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
#                          config.ROI_POSITIVE_RATIO)
#     positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
#     positive_count = tf.shape(positive_indices)[0]
#     # Negative ROIs. Add enough to maintain positive:negative ratio.
#     r = 1.0 / config.ROI_POSITIVE_RATIO
#     negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
#     negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
#     # Gather selected ROIs
#     positive_rois = tf.gather(proposals, positive_indices)
#     negative_rois = tf.gather(proposals, negative_indices)
#
#     # Assign positive ROIs to GT boxes.
#     positive_overlaps = tf.gather(overlaps, positive_indices)
#     roi_gt_box_assignment = tf.cond(
#         tf.greater(tf.shape(positive_overlaps)[1], 0),
#         true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
#         false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
#     )
#     roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
#     roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)
#
#     # Compute bbox refinement for positive ROIs
#     deltas = box_refinement_graph2(positive_rois, roi_gt_boxes, config)
#     # deltas /= config.BBOX_STD_DEV ######################################################
#
#     # Append negative ROIs and pad bbox deltas and masks that
#     # are not used for negative ROIs with zeros.
#     rois = tf.concat([positive_rois, negative_rois], axis=0)
#     N = tf.shape(negative_rois)[0]
#     P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
#     rois = tf.pad(rois, [(0, P), (0, 0)])
#     # roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
#     roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
#     deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
#
#     return rois, roi_gt_class_ids, deltas, positive_indices ##########################################


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [Batch_size, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
            Class-specific bbox refinements.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.squeeze(proposals, axis=0) ############################################
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")

    # # Handle COCO crowds
    # # A crowd box in COCO is a bounding box around several instances. Exclude
    # # them from training. A crowd box is given a negative class ID.
    # crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    # non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    # crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    # gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    # gt_boxes = tf.gather(gt_boxes, non_crowd_ix)

    # Compute overlaps matrix [proposals, gt_boxes]
    #############################################################
    # overlaps = overlaps_graph(proposals, gt_boxes)
    #
    overlaps = overlaps_graph(tf.round(proposals * config.IMG_SIZE[0]),
                              tf.round(gt_boxes * config.IMG_SIZE[0]))


    # Compute overlaps with crowd boxes [anchors, crowds]
    # crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    # crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    # no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine postive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max > 0.) ################################### >= 0.5
    positive_indices = tf.where(positive_roi_bool)[:, 0] #####################################################
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(roi_iou_max <= 0.)[:, 0] ################################## 0.5

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = box_refinement_graph2(tf.round(positive_rois * config.IMG_SIZE[0]),
                                   tf.round(roi_gt_boxes * config.IMG_SIZE[0]),
                                   config)
    # deltas /= config.BBOX_STD_DEV ######################################################

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    # roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])

    return rois, roi_gt_class_ids, deltas

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps

def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matricies of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros