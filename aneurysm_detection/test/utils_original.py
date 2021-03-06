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

def cal_result3(pred, label, one_hot=False, e=1e-6):
    ### for soft result ###
    # detect at least one pixel -> TP = 1
    # convert one-hot labels to multiple labels
    if one_hot:
        _pred = np.argmax(pred, axis=-1)
        _label = np.argmax(label, axis=-1)

    else:
        _pred = pred
        _label = label

    _pred = _pred.reshape(np.shape(_pred)[0], -1)
    _label = _label.reshape(np.shape(_label)[0], -1)

    TP, FP, FN, TN = 0., 0., 0., 0.

    for p, l in zip(_pred, _label):
        cm = confusion_matrix(l, p, labels=[0, 1])
        # TP = cm[1][1].astype(np.float32)
        # FP = cm[0][1].astype(np.float32)
        # FN = cm[1][0].astype(np.float32)
        # TN = cm[0][0].astype(np.float32)

        TP += 1. if (cm[1][1] != 0 and l.sum() != 0 and p.sum() != 0) else 0.
        FP += 1. if (cm[1][1] == 0 and l.sum() != 0 and p.sum() != 0) or (l.sum() == 0 and p.sum() != 0) else 0.
        FN += 1. if (l.sum() != 0 and p.sum() == 0) else 0.
        TN += 1. if (l.sum() == 0 and p.sum() == 0) else 0.

    mean_acc = (TP + TN + e) / (TP + FP + FN + TN + e)
    mean_sens = (TP + e) / (TP + FN + e)
    mean_spec = (TN + e) / (TN + FP + e)
    mean_miou = (TP + e) / (FP + FN + TP + e)
    mean_dice = (2 * TP + e) / (2 * TP + FP + FN + e)

    hdorff = max(directed_hausdorff(_pred, _label)[0], directed_hausdorff(_label, _pred)[0])

    return [mean_acc, mean_sens, mean_spec, mean_miou, mean_dice, hdorff]

def cal_result2(pred, label, one_hot=False, e=1e-6):
    # convert one-hot labels to multiple labels
    if one_hot:
        _pred = np.argmax(pred, axis=-1)
        _label = np.argmax(label, axis=-1)

    else:
        _pred = pred
        _label = label

    _pred = _pred.reshape(np.shape(_pred)[0], -1)
    _label = _label.reshape(np.shape(_label)[0], -1)
    acc_list = []
    sens_list = []
    spec_list = []
    miou_list = []
    dice_list = []

    for p, l in zip(_pred, _label):
        cm = confusion_matrix(l, p, labels=[0, 1])
        TP = cm[1][1].astype(np.float32)
        FP = cm[0][1].astype(np.float32)
        FN = cm[1][0].astype(np.float32)
        TN = cm[0][0].astype(np.float32)

        # accuracy, sensitivity, specificity, mean iou, dice coefficient, hausdorff
        acc = (TP + TN + e)  / (TP + FP + FN + TN + e)
        sens = (TP + e) / (TP + FN + e)
        spec = (TN + e) / (TN + FP + e)
        miou = (TP + e) / (FP + FN + TP + e)
        dice = (2 * TP + e)  / (2 * TP + FP + FN + e)

        acc_list.append(acc)
        sens_list.append(sens)
        spec_list.append(spec)
        miou_list.append(miou)
        dice_list.append(dice)

    mean_acc = np.mean(np.array(acc_list))
    mean_sens = np.mean(np.array(sens_list))
    mean_spec = np.mean(np.array(spec_list))
    mean_miou = np.mean(np.array(miou_list))
    mean_dice = np.mean(np.array(dice_list))

    hdorff = max(directed_hausdorff(_pred, _label)[0], directed_hausdorff(_label, _pred)[0])

    return [mean_acc, mean_sens, mean_spec, mean_miou, mean_dice, hdorff]

def cal_result(pred, label, one_hot=False, e=1e-6):
    # convert one-hot labels to multiple labels
    if one_hot:
        _pred = np.argmax(pred, axis=-1)
        _label = np.argmax(label, axis=-1)

    else:
        _pred = pred
        _label = label

    _pred1 = _pred.flatten()
    _label1 = _label.flatten()
    _pred2 = _pred.reshape(np.shape(_pred)[0], -1)
    _label2 = _label.reshape(np.shape(_label)[0], -1)

    cm = confusion_matrix(_label1, _pred1, labels=[0, 1])
    TP = cm[1][1].astype(np.float32)
    FP = cm[0][1].astype(np.float32)
    FN = cm[1][0].astype(np.float32)
    TN = cm[0][0].astype(np.float32)

    # accuracy, sensitivity, specificity, mean iou, dice coefficient, hausdorff
    acc = (TP + TN + e) / (TP + FP + FN + TN + e)
    sens = (TP + e) / (TP + FN + e)
    spec = (TN + e) / (TN + FP + e)
    miou = (TP + e) / (FP + FN + TP + e)
    dice = (2 * TP + e)  / (2 * TP + FP + FN + e)
    hdorff = max(directed_hausdorff(_pred2, _label2)[0], directed_hausdorff(_label2, _pred2)[0])

    return [acc, sens, spec, miou, dice, hdorff]

def convert_to_subregions(pred, label, convert_keys, one_hot=True):
    if one_hot:
        pred_arr = np.argmax(pred, axis=-1)
        label_arr = np.argmax(label, axis=-1)
    else:
        pred_arr = pred
        label_arr = label

    pred_list = []
    label_list = []

    _, pred_index = np.unique(pred_arr, return_inverse=True)
    _, label_index = np.unique(label_arr, return_inverse=True)

    for convert_key in convert_keys:
        key = np.array(convert_key)
        pred_list.append(key[pred_index].reshape(pred_arr.shape))
        label_list.append(key[label_index].reshape(label_arr.shape))

    return pred_list, label_list


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


# def focal_loss(output, target, epsilon=1e-6):
#     ndim = len(output.get_shape())
#     output /= tf.reduce_sum(output, axis=(ndim - 1), keep_dims=True)
#     output = tf.clip_by_value(output, epsilon, 1 - epsilon)
#     focal = -tf.reduce_mean(tf.square(tf.ones_like(output)-output)*target*tf.log(output), axis=tuple(range(ndim-1)))
#     #focal = -tf.reduce_sum(tf.square(tf.ones_like(output)-output)*target*tf.log(output), axis=tuple(range(ndim-1)))
#
#     return tf.reduce_sum(focal)


def focal_loss(prediction_tensor, target_tensor, alpha=0.25, gamma=2, epsilon=1e-6):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent) / (192 * 160 * 10)


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

def generalised_dice_loss(prediction,
                          ground_truth,
                          weight_map=None,
                          type_weight='Square'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017

    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)
    # if len(ground_truth.shape) == len(prediction.shape):
    #     ground_truth = ground_truth[..., -1]
    one_hot = ground_truth

    if weight_map is not None:
        n_classes = prediction.shape[1].value
        # weight_map_nclasses = tf.reshape(
        #     tf.tile(weight_map, [n_classes]), prediction.get_shape())
        weight_map_nclasses = tf.tile(
            tf.expand_dims(tf.reshape(weight_map, [-1]), 1), [1, n_classes])
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        # ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
        # intersect = tf.sparse_reduce_sum(one_hot * prediction,
        #                                  reduction_axes=[0])
        print('one_hot', one_hot)
        print('prediction', prediction)
        ref_vol = tf.reduce_sum(one_hot, [0])
        print('ref_vol', ref_vol)
        intersect = tf.reduce_sum(one_hot * prediction, [0])
        seg_vol = tf.reduce_sum(prediction, 0)
        print('seg_vol', seg_vol)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    # generalised_dice_denominator = \
    #     tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_denominator = tf.reduce_sum(
        tf.multiply(weights,  tf.maximum(seg_vol + ref_vol, 1)))
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0,
                                      generalised_dice_score)
    return 1 - generalised_dice_score

def wasserstein_disagreement_map(
        prediction, ground_truth, weight_map=None, M=None):
    """
    Function to calculate the pixel-wise Wasserstein distance between the
    flattened prediction and the flattened labels (ground_truth) with respect
    to the distance matrix on the label space M.

    :param prediction: the logits after softmax
    :param ground_truth: segmentation ground_truth
    :param M: distance matrix on the label space
    :return: the pixelwise distance map (wass_dis_map)
    """
    if weight_map is not None:
        # raise NotImplementedError
        tf.logging.warning('Weight map specified but not used.')

    assert M is not None, "Distance matrix is required."
    # pixel-wise Wassertein distance (W) between flat_pred_proba and flat_labels
    # wrt the distance matrix on the label space M
    n_classes = prediction.shape[-1].value
    ground_truth.set_shape(prediction.shape)
    unstack_labels = tf.unstack(ground_truth, axis=-1)
    unstack_labels = tf.cast(unstack_labels, dtype=tf.float64)
    unstack_pred = tf.unstack(prediction, axis=-1)
    unstack_pred = tf.cast(unstack_pred, dtype=tf.float64)
    # print("shape of M", M.shape, "unstacked labels", unstack_labels,
    #       "unstacked pred" ,unstack_pred)
    # W is a weighting sum of all pairwise correlations (pred_ci x labels_cj)
    pairwise_correlations = []
    for i in range(n_classes):
        for j in range(n_classes):
            pairwise_correlations.append(
                M[i, j] * tf.multiply(unstack_pred[i], unstack_labels[j]))
    wass_dis_map = tf.add_n(pairwise_correlations)
    return wass_dis_map

def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.

    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot

def generalised_wasserstein_dice_loss(prediction,
                                      ground_truth,
                                      weight_map=None):
    """
    Function to calculate the Generalised Wasserstein Dice Loss defined in

        Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score
        for Imbalanced Multi-class Segmentation using Holistic
        Convolutional Networks.MICCAI 2017 (BrainLes)

    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    M_tree = np.array([[0., 1., 1., 1., 1.],
                       [1., 0., 0.6, 0.2, 0.5],
                       [1., 0.6, 0., 0.6, 0.7],
                       [1., 0.2, 0.6, 0., 0.5],
                       [1., 0.5, 0.7, 0.5, 0.]], dtype=np.float64)

    if weight_map is not None:
        # raise NotImplementedError
        tf.logging.warning('Weight map specified but not used.')

    prediction = tf.cast(prediction, tf.float32)
    n_classes = prediction.shape[-1].value
    # one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])
    one_hot = ground_truth

    # labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])
    # one_hot = tf.sparse_tensor_to_dense(one_hot)

    # M = tf.cast(M, dtype=tf.float64)
    # compute disagreement map (delta)
    M = M_tree
    delta = wasserstein_disagreement_map(prediction, one_hot, M=M)
    print('delta', delta)
    # compute generalisation of all error for multi-class seg
    all_error = tf.reduce_sum(delta)
    print('all_error', all_error)
    # compute generalisation of true positives for multi-class seg
    one_hot = tf.cast(one_hot, dtype=tf.float64)
    print('one_hot', one_hot)
    true_pos = tf.reduce_sum(
        tf.multiply(tf.constant(M[0, :n_classes], dtype=tf.float64), one_hot),
        axis=-1)
    print('true_pos', true_pos)
    true_pos = tf.reduce_sum(tf.multiply(true_pos, 1. - delta), axis=0)
    print('true_pos2', true_pos)
    WGDL = 1. - (2. * true_pos) / (2. * true_pos + all_error)
    print('WGDL', WGDL)
    return tf.cast(WGDL, dtype=tf.float32)





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
    if mode == 'dice' or 'g_dice':
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

def select_downsampling(name, down_conv, down_pool, channel_n, pool_size_h, pool_size_w, mode):
    if mode == 'neighbor':
        shape = [-1, pool_size_h, pool_size_w, channel_n]
        down_pool = re_conv2D(name + '_neighbor', down_conv, shape)

    elif mode == 'maxpool':
        down_pool = maxpool(name + '_maxpool', down_conv, [2,2], [2,2], 'SAME')

    elif mode == 'avgpool':
        down_pool = averagepool(name + '_avgpool', down_conv, [2,2], [2,2], 'SAME')

    return down_pool

def select_downsampling2(name, down_conv, channel_n, pool_size_h, pool_size_w, mode):
    if mode == 'neighbor':
        shape = [-1, pool_size_h, pool_size_w, channel_n]
        down_pool = re_conv2D(name + '_neighbor', down_conv, shape)

    elif mode == 'maxpool':
        down_pool = maxpool(name + '_maxpool', down_conv, [2,2], [2,2], 'SAME')

    elif mode == 'avgpool':
        down_pool = averagepool(name + '_avgpool', down_conv, [2,2], [2,2], 'SAME')

    return down_pool


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


def select_upsampling2(name, up_conv, channel_n, pool_size_h, pool_size_w, mode, mul_channel=True):
    shape = [-1, pool_size_h, pool_size_w, channel_n]

    if mode == 'resize':
      up_pool = re_conv2D(name + '_reconv', up_conv, shape)

    elif mode == 'transpose':
        up_pool = deconv2D(name + 'deconv', up_conv, [3, 3, channel_n, channel_n * 2 if mul_channel else channel_n], shape, [1,2,2,1], 'SAME')
        up_pool = tf.reshape(up_pool, shape)

    elif mode == 'add':
        up_pool1 = re_conv2D(name + '_reconv', up_conv, shape)
        up_pool2 = deconv2D(name + 'deconv', up_conv, [3, 3, channel_n, channel_n * 2 if mul_channel else channel_n], shape, [1,2,2,1], 'SAME')
        up_pool2 = tf.reshape(up_pool2, shape)
        up_pool = up_pool1 + up_pool2

    elif mode == 'concat':
        up_pool1 = re_conv2D(name + '_reconv', up_conv, shape)
        up_pool2 = deconv2D(name + 'deconv', up_conv, [3, 3, channel_n, channel_n * 2 if mul_channel else channel_n], shape, [1, 2, 2, 1], 'SAME')
        up_pool2 = tf.reshape(up_pool2, shape)
        up_pool = concat(name + '_upsampling_concat', [up_pool1, up_pool2], axis=3)
        up_pool = conv2D(name + '_bottleneck', up_pool, channel_n, [1,1], [1,1], padding='SAME')

    elif mode == 'avgpool' :
        filter = tf.ones([2, 2, channel_n, channel_n * 2 if mul_channel else channel_n])  # [height, width, output_channels, in_channels]
        batch_size = tf.shape(up_conv)[0]
        shape2 = [batch_size, pool_size_h, pool_size_w, channel_n]
        up_pool = tf.nn.conv2d_transpose(up_conv, filter, shape2, [1, 2, 2, 1], 'SAME')

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

# Unet  (https://arxiv.org/abs/1505.04597)
def unet_down_block(inputs, conv_list, pool_list, channel_n, pool_size_h, pool_size_w, group_n, act_fn, norm_type, down_type, training, idx):
    conv_list[idx] = conv2D(str(idx) + '_downconv1', inputs, channel_n, [3, 3], [1, 1], padding='SAME')
    conv_list[idx] = Normalization(conv_list[idx], norm_type, training, str(idx) + '_downnorm1', G=group_n)
    conv_list[idx] = activation(str(idx) + '_downact1', conv_list[idx], act_fn)

    conv_list[idx] = conv2D(str(idx) + '_downconv2', conv_list[idx], channel_n, [3, 3], [1, 1], padding='SAME')
    conv_list[idx] = Normalization(conv_list[idx], norm_type, training, str(idx) + '_downnorm2', G=group_n)
    conv_list[idx] = activation(str(idx) + '_downact2', conv_list[idx], act_fn)
    print('down' + str(idx + 1) + 'conv', conv_list[idx])
    pool_list[idx] = select_downsampling(str(idx) + '_downsampling',
                                         conv_list[idx],
                                         pool_list[idx],
                                         channel_n,
                                         pool_size_h,
                                         pool_size_w,
                                         down_type)
    print('down' + str(idx + 1) + 'pool', pool_list[idx])

    if down_type == 'neighbor':
        conv_list[idx] = Normalization(conv_list[idx], norm_type, training,
                                             str(idx) + '_norm3', G=group_n)
        conv_list[idx] = activation(str(idx) + '_act3', conv_list[idx], act_fn)

    return pool_list[idx]


def unet_same_block(inputs, channel_n, group_n, act_fn, norm_type, training, name='same'):
    conv_list = conv2D(name + 'same_conv1', inputs, channel_n, [3, 3], [1, 1], padding='SAME')
    conv_list = Normalization(conv_list, norm_type, training, name + 'same_norm1', G=group_n)
    conv_list = activation(name + 'same_act1', conv_list, act_fn)
    conv_list = conv2D(name + 'same_conv2', conv_list, channel_n, [1, 1], [1, 1], padding='SAME')
    conv_list = Normalization(conv_list, norm_type, training, name + 'same_norm2', G=group_n)
    conv_list = activation(name + 'same_act2', conv_list, act_fn)

    return conv_list

def unet_up_block(inputs, downconv_list, upconv_list, pool_list, channel_n, group_n, act_fn, norm_type, training, idx, name='up'):
    pool_list[idx] = Normalization(inputs, norm_type, training, name + str(idx) + '_norm1', G=group_n)
    pool_list[idx] = activation(name + str(idx) + '_upsampling_act', pool_list[idx], act_fn)
    pool_list[idx] = concat(name + str(idx) + '_upconcat', [pool_list[idx], downconv_list[idx]], axis=3)

    upconv_list[idx] = conv2D(name + str(idx) + '_upconv1', pool_list[idx], channel_n, [3, 3], [1, 1], padding='SAME')
    upconv_list[idx] = Normalization(upconv_list[idx], norm_type, training, name + str(idx) + '_upnorm1', G=group_n)
    upconv_list[idx] = activation(name + str(idx) + '_upact1', upconv_list[idx], act_fn)

    upconv_list[idx] = conv2D(name + str(idx) + '_upconv2', upconv_list[idx], channel_n, [3, 3], [1, 1], padding='SAME')
    upconv_list[idx] = Normalization(upconv_list[idx], norm_type, training, name + str(idx) + '_upnorm2', G=group_n)
    upconv_list[idx] = activation(name + str(idx) + '_upact2', upconv_list[idx], act_fn)
    print('up' + str(idx + 1) + 'conv', upconv_list[idx])

    return upconv_list[idx]

def unet_up_block2(inputs, downconv_list, upconv_list, pool_list, channel_n, group_n, act_fn, norm_type, training, idx, n_layers=1, name='up'):
    pool_list[idx] = Normalization(inputs, norm_type, training, name + str(idx) + '_norm1', G=group_n)
    pool_list[idx] = activation(name + str(idx) + '_upsampling_act', pool_list[idx], act_fn)
    pool_list[idx] = concat(name + str(idx) + '_upconcat', [pool_list[idx], downconv_list[idx]], axis=3)

    for i in range(n_layers):
        upconv_list[idx] = conv2D(name + str(idx) + '_upconv1_' + str(i), pool_list[idx], channel_n, [3, 3], [1, 1], padding='SAME')
        upconv_list[idx] = Normalization(upconv_list[idx], norm_type, training, name + str(idx) + '_upnorm1_' + str(i), G=group_n)
        upconv_list[idx] = activation(name + str(idx) + '_upact1_' + str(i), upconv_list[idx], act_fn)

        upconv_list[idx] = conv2D(name + str(idx) + '_upconv2_' + str(i), upconv_list[idx], channel_n, [3, 3], [1, 1], padding='SAME')
        upconv_list[idx] = Normalization(upconv_list[idx], norm_type, training, name + str(idx) + '_upnorm2_' + str(i), G=group_n)
        upconv_list[idx] = activation(name + str(idx) + '_upact2_' + str(i), upconv_list[idx], act_fn)
        print('up' + str(idx + 1) + 'conv_' + str(i), upconv_list[idx])

    return upconv_list[idx]

# resnet  (https://arxiv.org/abs/1512.03385)
def residual_block_v1(inputs, channel_n, group_n, act_fn, norm_type, training, idx, shortcut=True):
    # input
    il = conv2D(str(idx) + '_input', inputs, channel_n, [1, 1], [1, 1], padding='SAME')
    il = Normalization(il, norm_type, training, str(idx) + '_input_norm', G=group_n)
    il = activation(str(idx) + '_input_act', il, act_fn)

    # bottleneck1
    hl = conv2D(str(idx) + '_bottleneck1', inputs, int(channel_n/4), [1, 1], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, str(idx) + '_bottleneck_norm1', G=group_n)
    hl = activation(str(idx) + '_bottleneck_act1', hl, act_fn)

    # conv
    hl = conv2D(str(idx) + '_conv', hl, int(channel_n / 4), [3, 3], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, str(idx) + '_conv_norm', G=group_n)
    hl = activation(str(idx) + '_conv_act', hl, act_fn)

    # bottleneck2
    hl = conv2D(str(idx) + '_bottleneck2', hl, channel_n, [1, 1], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, str(idx) + '_bottleneck_norm2', G=group_n)
    hl = activation(str(idx) + '_bottleneck_act2', hl, act_fn)

    hl = il + hl if shortcut else hl

    return hl

def residual_block_v1_dr(name, inputs, channel_n, group_n, drop_rate, act_fn, norm_type, training, idx, shortcut=True):
    # input
    il = conv2D(name + str(idx) + '_input', inputs, channel_n, [1, 1], [1, 1], padding='SAME')
    il = Normalization(il, norm_type, training, name + str(idx) + '_input_norm', G=group_n)
    il = activation(name + str(idx) + '_input_act', il, act_fn)

    # bottleneck1
    hl = conv2D(name + str(idx) + '_bottleneck1', inputs, int(channel_n/4), [1, 1], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, name + str(idx) + '_bottleneck_norm1', G=group_n)
    hl = activation(name + str(idx) + '_bottleneck_act1', hl, act_fn)
    # hl = dropout(name + str(idx) + '_dropout1', hl, drop_rate, training)

    # conv
    hl = conv2D(name + str(idx) + '_conv', hl, int(channel_n / 4), [3, 3], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, name + str(idx) + '_conv_norm', G=group_n)
    hl = activation(name + str(idx) + '_conv_act', hl, act_fn)
    hl = dropout(name + str(idx) + '_dropout2', hl, drop_rate, training)

    # bottleneck2
    hl = conv2D(name + str(idx) + '_bottleneck2', hl, channel_n, [1, 1], [1, 1], padding='SAME')
    hl = Normalization(hl, norm_type, training, name + str(idx) + '_bottleneck_norm2', G=group_n)
    hl = activation(name + str(idx) + '_bottleneck_act2', hl, act_fn)
    # hl = dropout(name + str(idx) + '_dropout3', hl, drop_rate, training)

    hl = il + hl if shortcut else hl

    return hl

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

# densenet  (https://arxiv.org/abs/1608.06993)
def dense_layer(name, inputs, group_n, drop_rate, act_fn, norm_type, growth, training, idx):
    # bottleneck
    l = Normalization(inputs, norm_type, training, name + str(idx) + '_bottleneck_norm1', G=group_n)
    l = activation(name + str(idx) + '_bottleneck_act1', l, act_fn)
    l = conv2D(name + str(idx) + '_bottleneck1', l, 4 * growth, [1, 1], [1, 1], 'SAME')
    l = dropout(name + str(idx) + '_dropout1', l, drop_rate, training)

    # conv
    l = Normalization(l, norm_type, training, name + str(idx) + '_bottleneck_norm2', G=group_n)
    l = activation(name + str(idx) + '_bottleneck_act2', l, act_fn)
    l = conv2D(name + str(idx) + '_bottleneck2', l, growth, [3, 3], [1, 1], 'SAME')
    l = dropout(name + str(idx) + '_dropout2', l, drop_rate, training)

    return l

def transition_layer(name, inputs, group_n, act_fn, norm_type, theta, training, specific_n_channels=False, idx=0):
    if specific_n_channels:
        l = Normalization(inputs, norm_type, training, name + str(idx) + '_bottleneck_norm1', G=group_n)
        l = activation(name + str(idx) + '_bottleneck_act1', l, act_fn)
        l = conv2D(name + str(idx) + '_bottleneck1', l, specific_n_channels, [1, 1], [1, 1], 'SAME')

    else:
        n_channels = inputs.get_shape().as_list()[-1]

        l = Normalization(inputs, norm_type, training, name + str(idx) + '_bottleneck_norm1', G=group_n)
        l = activation(name + str(idx) + '_bottleneck_act1', l, act_fn)
        l = conv2D(name + str(idx) + '_bottleneck1', l, n_channels * theta, [1, 1], [1, 1], padding='SAME')

    return l

def dense_block_v1(name, inputs, group_n, drop_rate, act_fn, norm_type, growth, training, n_layer):
    hl = tf.identity(inputs)

    for idx in range(n_layer):
        l = dense_layer(name = name,
                        inputs = hl,
                        group_n = group_n,
                        drop_rate = drop_rate,
                        act_fn = act_fn,
                        norm_type = norm_type,
                        growth = growth,
                        training = training,
                        idx = idx)
        hl = tf.concat([hl, l], axis=3)

    return hl

# depthwise separable conv  (Mobilenet - https://arxiv.org/abs/1704.04861)

def depthwise_separable_convlayer(name, inputs, channel_n, width_mul, group_n, act_fn, norm_type, training, idx, rate=None):
    # depthwise
    depthwise_filter = tf.get_variable(name=name+'depthwise_filter' + str(idx),
                                       shape=[3, 3, inputs.get_shape()[-1], width_mul],
                                       dtype=tf.float32,
                                       initializer=initializer)
    l = tf.nn.depthwise_conv2d(inputs, depthwise_filter, [1, 1, 1, 1], 'SAME', rate=rate, name = name + str(idx) + '_depthwise')
    l = Normalization(l, norm_type, training, name + str(idx) + '_depthwise_norm', G=group_n)
    l = activation(name + str(idx) + '_depthwise_act1', l, act_fn)

    # pointwise
    l = conv2D(name + str(idx) + '_pointwise', l, channel_n, [1, 1], [1, 1], padding='SAME')
    l = Normalization(l, norm_type, training, name + str(idx) + '_pointwise_norm1', G=group_n)
    l = activation(name + str(idx) + '_pointwise_act1', l, act_fn)

    return l

def depthwise_separable_convlayer2(name, inputs, channel_n, width_mul, group_n, act_fn, norm_type, training, idx, batch_size, threshold='fuzzy', n_divide=10, standard=False, scale=1, rate=None):
    # depthwise
    depthwise_filter = tf.get_variable(name=name+'depthwise_filter' + str(idx),
                                       shape=[3, 3, inputs.get_shape()[-1], width_mul],
                                       dtype=tf.float32,
                                       initializer=initializer)
    l = tf.nn.depthwise_conv2d(inputs, depthwise_filter, [1, 1, 1, 1], 'SAME', rate=rate, name = name + str(idx) + '_depthwise')
    # l = Normalization(l, norm_type, training, name + str(idx) + '_depthwise_norm', G=group_n)

    l = Normalization(l, 'batch_match', training, name + str(idx) + '_depthwise_norm', batch_size=batch_size,
                                 threshold=threshold, n_divide=n_divide, standard=standard, scale=scale, first=True)

    l = activation(name + str(idx) + '_depthwise_act1', l, act_fn)

    # pointwise
    l = conv2D(name + str(idx) + '_pointwise', l, channel_n, [1, 1], [1, 1], padding='SAME')
    # l = Normalization(l, norm_type, training, name + str(idx) + '_pointwise_norm1', G=group_n)
    l = Normalization(l, 'batch_match', training, name + str(idx) + 'pointwise_norm1', batch_size=batch_size,
                                 threshold=threshold, n_divide=n_divide, standard=standard, scale=scale, first=False)
    l = activation(name + str(idx) + '_pointwise_act1', l, act_fn)

    return l

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






# shufflenet  (https://arxiv.org/abs/1707.01083)

def channel_shuffle(name, inputs, group_n):
    with tf.variable_scope(name):
        # _, h, w, c = l.shape
        _, h, w, c = inputs.get_shape().as_list()
        _l = tf.reshape(inputs, [-1, h, w, group_n, c // group_n])
        _l = tf.transpose(_l, [0, 1, 2, 4, 3])
        l = tf.reshape(_l, [-1, h, w, c])

        return l

def group_conv2D(name, inputs, channel_n, kernel_size, group_n, stride, act_fn, norm_type, training, idx, use_act_fn=True, padding='SAME'):
    in_channel_per_group = inputs.get_shape().as_list()[-1] // group_n
    out_channel_per_group = channel_n // group_n
    grouped_channel_list = []

    for i in range(group_n):
        _l = conv2D(name = name + str(i),
                    inputs = inputs[:, :, :, i * in_channel_per_group: i * in_channel_per_group + in_channel_per_group],
                    filters = out_channel_per_group,
                    kernel_size = [kernel_size, kernel_size],
                    strides = [stride, stride],
                    padding = padding)
        grouped_channel_list.append(_l)

    _l = tf.concat(grouped_channel_list, axis=-1, name='concat_channel')
    _l = Normalization(_l, norm_type, training, name + str(idx) + 'norm', G=group_n)

    if use_act_fn:
        _l = activation(name + str(idx) + 'act', _l, act_fn)

    return _l

def shufflenet_unit(name, inputs, channel_n, group_n, stride, act_fn, norm_type, training, idx):
    if stride != 1:
        *_, c = inputs.get_shape().as_list()
        channel_n = channel_n - c # Residual 채널 수 c개 + Group 채널 수 = 최종 채널수(num_filter) 가 되어야 하므로

    # Residual part
    if stride != 1:
        residual_layer = averagepool('residual' + str(idx), inputs, [2,2], [2,2], 'SAME')
    else:
        residual_layer = tf.identity(inputs)

    # Group part
    depthwise_filter = tf.get_variable(name = name + 'depthwise_filter' + str(idx),
                                       shape = [3,3,inputs.get_shape()[-1], 1],
                                       dtype = tf.float32,
                                       initializer = tf.contrib.layers.variance_scaling_initializer())
    _group_layer = group_conv2D(name = name + '_group_conv1',
                                inputs = inputs,
                                channel_n = channel_n,
                                kernel_size = 1,
                                group_n = group_n,
                                stride = 1,
                                act_fn = act_fn,
                                norm_type = norm_type,
                                training = training,
                                idx = idx)

    _shuffled_group_layer = channel_shuffle(name = name + '_channel_shuffle',
                                            inputs = _group_layer,
                                            group_n = group_n)

    _depthwise_conv_layer = tf.nn.depthwise_conv2d(input = _shuffled_group_layer,
                                                   filter = depthwise_filter,
                                                   strides = [1,stride,stride,1],
                                                   padding = 'SAME',
                                                   name = name + str(idx) + '_depthwise')

    _depthwise_conv_layer = Normalization(x = _depthwise_conv_layer,
                                          norm_type = norm_type,
                                          is_train = training,
                                          name = name + str(idx) + 'depthwise_norm',
                                          G = group_n)

    final_group_layer = group_conv2D(name = name + '_group_conv2',
                                     inputs = _depthwise_conv_layer,
                                     channel_n = channel_n,
                                     kernel_size = 1,
                                     group_n = group_n,
                                     stride = 1,
                                     act_fn = act_fn,
                                     norm_type = norm_type,
                                     training = training,
                                     idx = idx,
                                     use_act_fn=False)

    # Concat part
    if stride != 1:
        layer = tf.concat([residual_layer, final_group_layer], axis=3)

    else:
        layer = residual_layer + final_group_layer

    final_layer = activation(name + str(idx) + '_shuffleunit_act', layer, act_fn)

    return final_layer

def shufflenet_stage(name, inputs, channel_n, group_n, act_fn, norm_type, training, repeat):
    l = shufflenet_unit(name = name,
                        inputs = inputs,
                        channel_n = channel_n,
                        group_n = group_n,
                        stride = 2,
                        act_fn = act_fn,
                        norm_type = norm_type,
                        training=training,
                        idx=0)

    for i in range(repeat):
        l = shufflenet_unit(name = name + str(i),
                            inputs = inputs,
                            channel_n = channel_n,
                            group_n = group_n,
                            stride = 1,
                            act_fn = act_fn,
                            norm_type = norm_type,
                            training=training,
                            idx=i+1)

    return l


# HENet  (https://arxiv.org/abs/1803.02742)
def he_hlayer(name, inputs, channel_n, group_m, group_n, act_fn, norm_type, training, idx):
    l = group_conv2D(name = name + str(idx) + '_1st_Gconv',
                     inputs = inputs,
                     channel_n = channel_n,
                     kernel_size = 1,
                     group_n = group_m,
                     stride = 1,
                     act_fn = act_fn,
                     norm_type = norm_type,
                     training = training,
                     idx = 0,
                     use_act_fn = False)

    l = channel_shuffle(name = name + str(idx) + '_channelshuffle',
                        inputs = l,
                        group_n = group_m)

    l = group_conv2D(name = name + str(idx) + '_2nd_Gconv',
                     inputs = l,
                     channel_n = channel_n,
                     kernel_size = 3,
                     group_n = group_n,
                     stride = 1,
                     act_fn = act_fn,
                     norm_type = norm_type,
                     training = training,
                     idx = 1,
                     use_act_fn = True)
    return l

def he_s1block(name, inputs, channel_n, group_m, group_n, act_fn, norm_type, training, repeat):
    X = tf.identity(inputs)
    l = inputs
    for i in range(repeat):
        HL = he_hlayer(name = name + '_hlayer{}'.format(i + 1),
                       inputs = l,
                       channel_n = channel_n,
                       group_m = group_m,
                       group_n = group_n,
                       act_fn = act_fn,
                       norm_type = norm_type,
                       training = training,
                       idx = i)
        X = X + HL
        l = tf.concat([HL, X], axis=3)

    return l

def he_s2block(name, inputs, channel_n, group_m, group_n, act_fn, norm_type, training, resize = True):
    l = group_conv2D(name = name + '_1st_s2_Gconv',
                     inputs = inputs,
                     channel_n = channel_n // 2,
                     kernel_size = 3,
                     group_n = group_m,
                     stride = 2 if resize else 1,
                     act_fn = act_fn,
                     norm_type = norm_type,
                     training = training,
                     idx = 0,
                     use_act_fn = False)

    l = channel_shuffle(name = name + '_s2_channelshuffle',
                        inputs = l,
                        group_n = group_m)

    l = group_conv2D(name = name + '_2nd_s2_Gconv',
                     inputs = l,
                     channel_n = channel_n,
                     kernel_size = 1,
                     group_n = group_n,
                     stride = 1,
                     act_fn = act_fn,
                     norm_type = norm_type,
                     training = training,
                     idx = 1,
                     use_act_fn = True)
    return l

def he_stage(name, inputs, channel_in, channel_out, group_m, group_n, act_fn, norm_type, training, repeat, resize=True, last_stage=False):
    if last_stage:
        _, h, _, _ = inputs.get_shape().as_list()

        l = group_conv2D(name = name + 'last_stage1',
                         inputs = inputs,
                         channel_n = channel_in,
                         kernel_size = h,
                         group_n = group_m,
                         stride = 1,
                         act_fn = act_fn,
                         norm_type = norm_type,
                         training = training,
                         idx = 0,
                         use_act_fn = False,
                         padding = 'VALID')

        l = channel_shuffle(name = name + '_last_channelshuffle',
                            inputs = l,
                            group_n = group_m)

        l = group_conv2D(name = name + 'last_stage2',
                         inputs = l,
                         channel_n = channel_out,
                         kernel_size = 1,
                         group_n = group_n,
                         stride = 1,
                         act_fn = act_fn,
                         norm_type = norm_type,
                         training = training,
                         idx = 1,
                         use_act_fn = True)
    else:
        l = he_s1block(name = name + 's1_block',
                       inputs = inputs,
                       channel_n = channel_in,
                       group_m = group_m,
                       group_n = group_n,
                       act_fn = act_fn,
                       norm_type = norm_type,
                       training = training,
                       repeat = repeat)

        l = he_s2block(name = name + 's2_block',
                       inputs = l,
                       channel_n = channel_out,
                       group_m = group_m,
                       group_n = group_n,
                       act_fn = act_fn,
                       norm_type = norm_type,
                       training = training,
                       resize = resize)

    return l
################################################################################################

def atrous_spatial_pyramid_pooling(name, inputs, channel_n, atrous_rate_list, act_fn, training):
    atrous_rates = atrous_rate_list

    ### a) 1x1 Conv * 1  +  3x3 Conv * 3
    conv_1x1 = conv2D(name + '_a_1x1', inputs, channel_n, [1,1], [1,1], padding='SAME')
    conv_1x1 = Normalization(conv_1x1, 'batch', training, name + '_a_1x1_norm')
    conv_1x1 = activation(name + '_a_1x1_act', conv_1x1, act_fn)

    conv_3x3_0 = conv2D(name + '_a_3x3_0', inputs, channel_n, [3,3], [1,1], dilation_rate=atrous_rates[0], padding='SAME')
    conv_3x3_0 = Normalization(conv_3x3_0, 'batch', training, name + '_a_3x3_0_norm')
    conv_3x3_0 = activation(name + 'a_3x3_0_act', conv_3x3_0, act_fn)

    conv_3x3_1 = conv2D(name + '_a_3x3_1', inputs, channel_n, [3, 3], [1, 1], dilation_rate=atrous_rates[1], padding='SAME')
    conv_3x3_1 = Normalization(conv_3x3_1, 'batch', training, name + '_a_3x3_1_norm')
    conv_3x3_1 = activation(name + 'a_3x3_1_act', conv_3x3_1, act_fn)

    conv_3x3_2 = conv2D(name + '_a_3x3_2', inputs, channel_n, [3, 3], [1, 1], dilation_rate=atrous_rates[2], padding='SAME')
    conv_3x3_2 = Normalization(conv_3x3_2, 'batch', training, name + '_a_3x3_2_norm')
    conv_3x3_2 = activation(name + 'a_3x3_2_act', conv_3x3_2, act_fn)

    ### (b) the image-level features
    # global average pooling
    img_lv_features = GlobalAveragePooling2D(inputs, channel_n, name + '_GAP', keep_dims=True)
    # 1x1 conv
    img_lv_features = conv2D(name + '_img_lv_features', img_lv_features, channel_n, [1,1], [1,1], padding='SAME')
    img_lv_features = Normalization(img_lv_features, 'batch', training, name + '_img_lv_features_norm')
    img_lv_features = activation(name + '_img_lv_features_act', img_lv_features, act_fn)
    # upsampling
    img_lv_features = re_conv2D(name + '_upsample', img_lv_features, [-1, tf.shape(inputs)[1], tf.shape(inputs)[2], channel_n])
    # concat
    aspp_layer = tf.concat([conv_1x1, conv_3x3_0, conv_3x3_1, conv_3x3_2, img_lv_features], axis=3, name=name+'_concat')
    # 1x1 conv
    aspp_layer = conv2D(name + '_aspp_layer', aspp_layer, channel_n, [1,1], [1,1], padding='SAME')
    aspp_layer = Normalization(aspp_layer, 'batch', training, name + '_aspp_layer_norm')
    aspp_layer = activation(name + '_aspp_layer_act', aspp_layer, act_fn)

    return aspp_layer


def atrous_spatial_pyramid_pooling2(name, inputs, channel_n, atrous_rate_list, act_fn, training, batch_size, threshold='fuzzy', n_divide=10, standard=False, scale=1):
    atrous_rates = atrous_rate_list

    ### a) 1x1 Conv * 1  +  3x3 Conv * 3
    conv_1x1 = conv2D(name + '_a_1x1', inputs, channel_n, [1,1], [1,1], padding='SAME')
    # conv_1x1 = Normalization(conv_1x1, 'batch', training, name + '_a_1x1_norm')

    conv_1x1 = Normalization(conv_1x1, 'batch_match', training, name + '_a_1x1_norm', batch_size=batch_size,
                             threshold=threshold, n_divide=n_divide, standard=standard, scale=scale, first=True)

    conv_1x1 = activation(name + '_a_1x1_act', conv_1x1, act_fn)

    conv_3x3_0 = conv2D(name + '_a_3x3_0', inputs, channel_n, [3,3], [1,1], dilation_rate=atrous_rates[0], padding='SAME')
    # conv_3x3_0 = Normalization(conv_3x3_0, 'batch', training, name + '_a_3x3_0_norm')
    conv_3x3_0 = Normalization(conv_3x3_0, 'batch_match', training, name + '_a_3x3_0_norm', batch_size=batch_size,
                             threshold=threshold, n_divide=n_divide, standard=standard, scale=scale, first=True)
    conv_3x3_0 = activation(name + 'a_3x3_0_act', conv_3x3_0, act_fn)

    conv_3x3_1 = conv2D(name + '_a_3x3_1', inputs, channel_n, [3, 3], [1, 1], dilation_rate=atrous_rates[1], padding='SAME')
    # conv_3x3_1 = Normalization(conv_3x3_1, 'batch', training, name + '_a_3x3_1_norm')
    conv_3x3_1 = Normalization(conv_3x3_1, 'batch_match', training, name + '_a_3x3_1_norm', batch_size=batch_size,
                               threshold=threshold, n_divide=n_divide, standard=standard, scale=scale, first=True)
    conv_3x3_1 = activation(name + 'a_3x3_1_act', conv_3x3_1, act_fn)

    conv_3x3_2 = conv2D(name + '_a_3x3_2', inputs, channel_n, [3, 3], [1, 1], dilation_rate=atrous_rates[2], padding='SAME')
    # conv_3x3_2 = Normalization(conv_3x3_2, 'batch', training, name + '_a_3x3_2_norm')
    conv_3x3_2 = Normalization(conv_3x3_2, 'batch_match', training, name + '_a_3x3_2_norm', batch_size=batch_size,
                               threshold=threshold, n_divide=n_divide, standard=standard, scale=scale, first=True)
    conv_3x3_2 = activation(name + 'a_3x3_2_act', conv_3x3_2, act_fn)

    ### (b) the image-level features
    # global average pooling
    img_lv_features = GlobalAveragePooling2D(inputs, channel_n, name + '_GAP', keep_dims=True)
    # 1x1 conv
    img_lv_features = conv2D(name + '_img_lv_features', img_lv_features, channel_n, [1,1], [1,1], padding='SAME')
    # img_lv_features = Normalization(img_lv_features, 'batch', training, name + '_img_lv_features_norm')
    img_lv_features = Normalization(img_lv_features, 'batch_match', training, name + '_img_lv_features_norm', batch_size=batch_size,
                               threshold=threshold, n_divide=n_divide, standard=standard, scale=scale, first=True)

    img_lv_features = activation(name + '_img_lv_features_act', img_lv_features, act_fn)
    # upsampling
    img_lv_features = re_conv2D(name + '_upsample', img_lv_features, [-1, tf.shape(inputs)[1], tf.shape(inputs)[2], channel_n])
    # concat
    aspp_layer = tf.concat([conv_1x1, conv_3x3_0, conv_3x3_1, conv_3x3_2, img_lv_features], axis=3, name=name+'_concat')
    # 1x1 conv
    aspp_layer = conv2D(name + '_aspp_layer', aspp_layer, channel_n, [1,1], [1,1], padding='SAME')
    # aspp_layer = Normalization(aspp_layer, 'batch', training, name + '_aspp_layer_norm')
    aspp_layer = Normalization(aspp_layer, 'batch_match', training, name + '_aspp_layer_norm', batch_size=batch_size,
                               threshold=threshold, n_divide=n_divide, standard=standard, scale=scale, first=True)
    aspp_layer = activation(name + '_aspp_layer_act', aspp_layer, act_fn)

    return aspp_layer

def atrous_spatial_pyramid_pooling3(name, inputs, channel_n, output_stride, act_fn, training):
    if output_stride not in [8, 16]:
        raise ValueError('output_stride must be in 8 or 16')
    multi_grid = [1,2,3]
    atrous_rates = [grid * (6 if output_stride == 16 else 12) for grid in multi_grid]

    ### a) 1x1 Conv * 1  +  3x3 Conv * 3
    conv_1x1 = conv2D(name + '_a_1x1', inputs, channel_n, [1,1], [1,1], padding='SAME')
    conv_1x1 = Normalization(conv_1x1, 'batch', training, name + '_a_1x1_norm')
    conv_1x1 = activation(name + '_a_1x1_act', conv_1x1, act_fn)

    conv_3x3_0 = conv2D(name + '_a_3x3_0', inputs, channel_n, [3,3], [1,1], dilation_rate=atrous_rates[0], padding='SAME')
    conv_3x3_0 = Normalization(conv_3x3_0, 'batch', training, name + '_a_3x3_0_norm')
    conv_3x3_0 = activation(name + 'a_3x3_0_act', conv_3x3_0, act_fn)

    conv_3x3_1 = conv2D(name + '_a_3x3_1', inputs, channel_n, [3, 3], [1, 1], dilation_rate=atrous_rates[1], padding='SAME')
    conv_3x3_1 = Normalization(conv_3x3_1, 'batch', training, name + '_a_3x3_1_norm')
    conv_3x3_1 = activation(name + 'a_3x3_1_act', conv_3x3_1, act_fn)

    conv_3x3_2 = conv2D(name + '_a_3x3_2', inputs, channel_n, [3, 3], [1, 1], dilation_rate=atrous_rates[2], padding='SAME')
    conv_3x3_2 = Normalization(conv_3x3_2, 'batch', training, name + '_a_3x3_2_norm')
    conv_3x3_2 = activation(name + 'a_3x3_2_act', conv_3x3_2, act_fn)

    ### (b) the image-level features
    # global average pooling
    img_lv_features = GlobalAveragePooling2D(inputs, channel_n, name + '_GAP', keep_dims=True)
    # 1x1 conv
    img_lv_features = conv2D(name + '_img_lv_features', img_lv_features, channel_n, [1,1], [1,1], padding='SAME')
    img_lv_features = Normalization(img_lv_features, 'batch', training, name + '_img_lv_features_norm')
    img_lv_features = activation(name + '_img_lv_features_act', img_lv_features, act_fn)
    # upsampling
    img_lv_features = tf.image.resize_bilinear(img_lv_features, tf.shape(inputs)[1:3], name=name + '_upsample')
    # concat
    aspp_layer = tf.concat([conv_1x1, conv_3x3_0, conv_3x3_1, conv_3x3_2, img_lv_features], axis=3, name=name+'_concat')
    # 1x1 conv
    aspp_layer = conv2D(name + '_aspp_layer', aspp_layer, channel_n, [1,1], [1,1], padding='SAME')
    aspp_layer = Normalization(aspp_layer, 'batch', training, name + '_aspp_layer_norm')
    aspp_layer = activation(name + '_aspp_layer_act', aspp_layer, act_fn)

    return aspp_layer

def xception_depthwise_separable_convlayer(name, inputs, channel_n, last_stride, act_fn, training, shortcut_conv=False, atrous=False, atrous_rate=2):
    rate = [atrous_rate, atrous_rate] if atrous else None
    # shortcut layer
    shortcut = tf.identity(inputs)

    if shortcut_conv:
        shortcut = conv2D(name + '_shortcut', shortcut, channel_n, [1,1], [last_stride, last_stride], padding='SAME')
        shortcut = Normalization(shortcut, 'batch', training, name + '_shortcut_norm')
        shortcut = activation(name + '_shortcut_act', shortcut, act_fn)

    in_channel = inputs.get_shape().as_list()[-1]
    width_mul = int(channel_n / in_channel)

    depthwise_filter1 = tf.get_variable(name = name + '_depthwise_filter1',
                                        shape = [3, 3, in_channel, width_mul],
                                        dtype = tf.float32,
                                        initializer = initializer)

    depthwise_filter2 = tf.get_variable(name = name + '_depthwise_filter2',
                                        shape = [3, 3, channel_n, 1],
                                        dtype = tf.float32,
                                        initializer = initializer)
    # conv layer 1
    l = tf.nn.depthwise_conv2d(inputs, depthwise_filter1, [1, 1, 1, 1], 'SAME', rate = None, name = name + '_sep1')
    l = Normalization(l, 'batch', training, name + '_sep_norm1')
    l = activation(name + '_sep_act1', l, act_fn)

    # conv layer 2
    l = tf.nn.depthwise_conv2d(l, depthwise_filter2, [1, 1, 1, 1], 'SAME', rate = None, name = name + '_sep2')
    l = Normalization(l, 'batch', training, name + '_sep_norm2')
    l = activation(name + '_sep_act2', l, act_fn)

    # conv layer 3
    l = tf.nn.depthwise_conv2d(l, depthwise_filter2, [1, last_stride, last_stride, 1], 'SAME', rate = rate, name = name + '_sep3')
    l = Normalization(l, 'batch', training, name + '_sep_norm3')
    l = activation(name + '_sep_act3', l, act_fn)

    # add layer
    l = l + shortcut

    return l

def xception_depthwise_separable_convlayer_dr(name, inputs, channel_n, last_stride, drop_rate, act_fn, training, shortcut_conv=False, atrous=False, atrous_rate=2):
    rate = [atrous_rate, atrous_rate] if atrous else None
    # shortcut layer
    shortcut = tf.identity(inputs)

    if shortcut_conv:
        shortcut = conv2D(name + '_shortcut', shortcut, channel_n, [1,1], [last_stride, last_stride], padding='SAME')
        shortcut = Normalization(shortcut, 'batch', training, name + '_shortcut_norm')
        shortcut = activation(name + '_shortcut_act', shortcut, act_fn)

    in_channel = inputs.get_shape().as_list()[-1]
    width_mul = int(channel_n / in_channel)

    depthwise_filter1 = tf.get_variable(name = name + '_depthwise_filter1',
                                        shape = [3, 3, in_channel, width_mul],
                                        dtype = tf.float32,
                                        initializer = initializer)

    depthwise_filter2 = tf.get_variable(name = name + '_depthwise_filter2',
                                        shape = [3, 3, channel_n, 1],
                                        dtype = tf.float32,
                                        initializer = initializer)
    # conv layer 1
    l = tf.nn.depthwise_conv2d(inputs, depthwise_filter1, [1, 1, 1, 1], 'SAME', rate = None, name = name + '_sep1')
    l = Normalization(l, 'batch', training, name + '_sep_norm1')
    l = activation(name + '_sep_act1', l, act_fn)
    l = dropout(name + '_dropout1', l, drop_rate, training)

    # conv layer 2
    l = tf.nn.depthwise_conv2d(l, depthwise_filter2, [1, 1, 1, 1], 'SAME', rate = None, name = name + '_sep2')
    l = Normalization(l, 'batch', training, name + '_sep_norm2')
    l = activation(name + '_sep_act2', l, act_fn)
    l = dropout(name + '_dropout1', l, drop_rate, training)

    # conv layer 3
    l = tf.nn.depthwise_conv2d(l, depthwise_filter2, [1, last_stride, last_stride, 1], 'SAME', rate = rate, name = name + '_sep3')
    l = Normalization(l, 'batch', training, name + '_sep_norm3')
    l = activation(name + '_sep_act3', l, act_fn)
    l = dropout(name + '_dropout1', l, drop_rate, training)

    # add layer
    l = l + shortcut

    return l



def xception_depthwise_separable_convlayer2(name, inputs, channel_n, last_stride, act_fn, training, batch_size, threshold = 'fuzzy',n_divide = 4,standard=False, scale=1, shortcut_conv=False, atrous=False, atrous_rate=2):
    rate = [atrous_rate, atrous_rate] if atrous else None
    # shortcut layer
    shortcut = tf.identity(inputs)

    if shortcut_conv:
        shortcut = conv2D(name + '_shortcut', shortcut, channel_n, [1,1], [last_stride, last_stride], padding='SAME')
        shortcut = Normalization(shortcut, 'batch_match', training, name + '_shortcut_norm', batch_size=batch_size,
                                 threshold=threshold, n_divide=n_divide, standard=standard, scale=scale, first=True)
        # il = Normalization(il, norm_type, training, name + str(idx) + '_input_norm', G=group_n, batch_size=batch_size,
        #                    threshold=threshold, n_divide=n_divide, standard=standard, scale=scale, first=True)

        shortcut = activation(name + '_shortcut_act', shortcut, act_fn)

    in_channel = inputs.get_shape().as_list()[-1]
    width_mul = int(channel_n / in_channel)

    depthwise_filter1 = tf.get_variable(name = name + '_depthwise_filter1',
                                        shape = [3, 3, in_channel, width_mul],
                                        dtype = tf.float32,
                                        initializer = initializer)

    depthwise_filter2 = tf.get_variable(name = name + '_depthwise_filter2',
                                        shape = [3, 3, channel_n, 1],
                                        dtype = tf.float32,
                                        initializer = initializer)
    # conv layer 1
    l = tf.nn.depthwise_conv2d(inputs, depthwise_filter1, [1, 1, 1, 1], 'SAME', rate = None, name = name + '_sep1')
    # l = Normalization(l, 'batch', training, name + '_sep_norm1')
    l = Normalization(shortcut, 'batch_match', training, name + '_sep_norm1', batch_size=batch_size,
                                 threshold=threshold, n_divide=n_divide, standard=standard, scale=scale, first=True)
    l = activation(name + '_sep_act1', l, act_fn)

    # conv layer 2
    l = tf.nn.depthwise_conv2d(l, depthwise_filter2, [1, 1, 1, 1], 'SAME', rate = None, name = name + '_sep2')
    # l = Normalization(l, 'batch', training, name + '_sep_norm2')
    l = Normalization(shortcut, 'batch_match', training, name + '_sep_norm2', batch_size=batch_size,
                                 threshold=threshold, n_divide=n_divide, standard=standard, scale=scale, first=False)
    l = activation(name + '_sep_act2', l, act_fn)

    # conv layer 3
    l = tf.nn.depthwise_conv2d(l, depthwise_filter2, [1, last_stride, last_stride, 1], 'SAME', rate = rate, name = name + '_sep3')
    # l = Normalization(l, 'batch', training, name + '_sep_norm3')
    l = Normalization(shortcut, 'batch_match', training, name + '_sep_norm3', batch_size=batch_size,
                                 threshold=threshold, n_divide=n_divide, standard=standard, scale=scale, first=False)
    l = activation(name + '_sep_act3', l, act_fn)

    # add layer
    l = l + shortcut

    return l

def xception_depthwise_separable_convlayer3(name, inputs, channel_n, last_stride, act_fn, training, shortcut_conv=False, atrous=False):
    rate = [[1, 1], [2, 2], [4, 4]] if atrous else [None, None, None]
    # shortcut layer
    shortcut = tf.identity(inputs)
    if shortcut_conv:
        shortcut = conv2D(name + '_shortcut', shortcut, channel_n, [1,1], [last_stride, last_stride], padding='SAME')
        shortcut = Normalization(shortcut, 'batch', training, name + '_shortcut_norm')
        shortcut = activation(name + '_shortcut_act', shortcut, act_fn)

    in_channel = inputs.get_shape().as_list()[-1]
    width_mul = int(channel_n / in_channel)

    depthwise_filter1 = tf.get_variable(name = name + '_depthwise_filter1',
                                        shape = [3, 3, in_channel, width_mul],
                                        dtype = tf.float32,
                                        initializer = initializer)

    depthwise_filter2 = tf.get_variable(name = name + '_depthwise_filter2',
                                        shape = [3, 3, channel_n, 1],
                                        dtype = tf.float32,
                                        initializer = initializer)
    # conv layer 1
    l = tf.nn.depthwise_conv2d(inputs, depthwise_filter1, [1,1,1,1], 'SAME', rate = rate[0], name = name + '_sep1')
    l = Normalization(l, 'batch', training, name + '_sep_norm1')
    l = activation(name + '_sep_act1', l, act_fn)

    # conv layer 2
    l = tf.nn.depthwise_conv2d(l, depthwise_filter2, [1, 1, 1, 1], 'SAME', rate = rate[1], name = name + '_sep2')
    l = Normalization(l, 'batch', training, name + '_sep_norm2')
    l = activation(name + '_sep_act2', l, act_fn)

    # conv layer 3
    l = tf.nn.depthwise_conv2d(l, depthwise_filter2, [1, last_stride, last_stride, 1], 'SAME', rate = rate[2], name = name + '_sep3')
    l = Normalization(l, 'batch', training, name + '_sep_norm3')
    l = activation(name + '_sep_act3', l, act_fn)

    # add layer
    l = l + shortcut

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


def rgb2gray(rgb):
    '''
    Change rgb to grayscale image
    '''
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_ND_bounding_box(input, margin):
    '''
    get bounding box excluding background
    :return: min, max indexes, and width, length, height of bounding box
    '''
    input_shape = input.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indices = np.nonzero(input)
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indices[i].min())
        idx_max.append(indices[i].max())
    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    width = idx_max[0] - idx_min[0]
    length = idx_max[1] - idx_min[1]
    height = idx_max[2] - idx_min[2]
    return idx_min, idx_max, width, length, height

def crop_volume_with_bounding_box(volume, min_idx, max_idx):
    output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    return output

def get_area(label):
    indices = np.nonzero(label)
    area = len(indices[0])
    return area

def get_volume(label):
    indices = np.nonzero(label)
    vol = len(indices[0])
    return vol

def get_glcm(input):
    '''
    get gray level co-occurrence matrix and then it's features
    '''
    glcm = greycomatrix(input, [1], [0], 256, normed=True)
    cont = greycoprops(glcm, 'contrast')
    diss = greycoprops(glcm, 'dissimilarity')
    homo = greycoprops(glcm, 'homogeneity')
    eng = greycoprops(glcm, 'energy')
    corr = greycoprops(glcm, 'correlation')
    ASM = greycoprops(glcm, 'ASM')
    return [cont, diss, homo, eng, corr, ASM]

def save_array_as_nifty_volume(data, filename, reference_name = None):
    '''
    :param data: np array
    :param filename: path + filename want to save
    '''
    if data.ndim == 3:
        transposed_data = np.transpose(data, [2,1,0])
    if data.ndim == 2:
        transposed_data = np.transpose(data, [1,0])
    img = sitk.GetImageFromArray(transposed_data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)

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

def add_values_to_key(target_list):
    '''
    marge several dictionaries with same key in list to one dictionaries
    :return:
    '''
    subdict = {}
    for i in range(len(target_list)):
        for k,v in target_list[i].items():
            subdict.setdefault(k, [])
            subdict[k].append(v)
    for k,v in subdict.items():
        subdict[k] = tuple(subdict[k])
    return subdict

def drop_col_contains_sth(dataframe,sth):
    '''
    drop columns contains certain strings(sth) in the dataframe
    :param sth:  str 'sth'
    :return:
    '''
    dropped_df = dataframe[dataframe.columns.drop(list(dataframe.filter(regex=sth)))]
    return dropped_df

def get_path_list(data_path):
    '''
    :return: all folder(or file list)
    '''
    id_list = []
    for path in data_path:
        path_list = tl.files.load_folder_list(path)
        id_list += [os.path.join(path, os.path.basename(p), os.path.basename(p)) for p in path_list]
    return id_list

def extract_patches_from_batch(imgs, patch_shape, stride):
    # simple version of sklearn.feature_extraction.image.extract_patches

    # if input imgs are not multiple imgs(just one img), then add axis=0 to make shape like [batch_size, w, h, ...]
    if imgs.ndim == 2 or (imgs.ndim == 3 and len(patch_shape) == 3):
        imgs = np.expand_dims(imgs, axis=0)

    patch_shape = (len(imgs),) + patch_shape
    patch_transpose = (3,0,1,2,4,5) if len(patch_shape) == 3 else (4,0,1,2,3,5,6,7)
    patch_reshape = (-1,) + patch_shape[1:]
    patch = image.extract_patches(imgs, patch_shape, extraction_step=stride)

    return patch.transpose(patch_transpose).reshape(patch_reshape)

def reconstruct_from_patches_nd(patches, image_shape, stride):
    # modified version of sklearn.feature_extraction.image.reconstruct_from_patches_2d
    # It can make only one image
    assert len(image_shape) < 4, 'image_shape must be [image_height, image_width, image_depth]'

    if len(stride) == 1 :
        i_h, i_w = image_shape[:2]
        p_h, p_w = patches.shape[1:3]
        img = np.zeros(image_shape)
        img_overlapped = np.zeros(image_shape)

        n_h = i_h - p_h + 1
        n_w = i_w - p_w + 1
        for p, (i, j) in zip(patches, product(range(0,n_h,stride), range(0,n_w,stride))):
            if patches.ndim == 3:
                img[i:i + p_h, j:j + p_w] += p
                img_overlapped[i:i + p_h, j:j + p_w] += 1
            elif patches.ndim == 4:
                img[i:i + p_h, j:j + p_w, :] += p
                img_overlapped[i:i + p_h, j:j + p_w, :] += 1

    else :
        assert len(stride) == 4, 'stride.shape must be stride or [1, x_stride, y_stride, z_stride] '
        i_h, i_w, i_d = image_shape
        p_h, p_w, p_d = patches.shape[1:]
        img = np.zeros(image_shape)
        img_overlapped = np.zeros(image_shape)

        n_h = i_h - p_h + 1
        n_w = i_w - p_w + 1
        n_d = i_d - p_d + 1

        for p, (i, j, k) in zip(patches, product(range(0,n_h,stride[1]), range(0,n_w,stride[2]), range(0,n_d,stride[3]))):
            img[i:i + p_h, j:j + p_w, k:k + p_d] += p
            img_overlapped[i:i + p_h, j:j + p_w, k:k + p_d] += 1


        img /= img_overlapped

    return img


# def reconstruct_from_patches_nd(patches, image_shape, stride):
#     # modified version of sklearn.feature_extraction.image.reconstruct_from_patches_2d
#     # It can make only one image
#     i_h, i_w = image_shape[:2]
#     p_h, p_w = patches.shape[1:3]
#     img = np.zeros(image_shape)
#     img_overlapped = np.zeros(image_shape)
#
#     n_h = i_h - p_h + 1
#     n_w = i_w - p_w + 1
#     for p, (i, j) in zip(patches, product(range(0,n_h,stride), range(0,n_w,stride))):
#         if patches.ndim == 3:
#             img[i:i + p_h, j:j + p_w] += p
#             img_overlapped[i:i + p_h, j:j + p_w] += 1
#         elif patches.ndim == 4:
#             img[i:i + p_h, j:j + p_w, :] += p
#             img_overlapped[i:i + p_h, j:j + p_w, :] += 1
#
#
#     img /= img_overlapped
#
#     return img

def discard_patch_idx(input, cut_line):
    n_non_zero = np.count_nonzero(input, axis=tuple(i for i in range(input.ndim) if not i == 0)) / np.prod(input.shape[1:])
    return np.where(n_non_zero >= cut_line)



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
#                                        Tensor Histogram Match                                                             #
#############################################################################################################################

def cal_hm_landmark_tensor(arr, max_percent = 99.8, threshold = 'fuzzy', n_divide = 4, standard=False, scale=1):
    if arr.get_shape().ndims > 1:
        arr = tf.reshape(arr, [-1])
    # if tf.errors.InvalidArgumentError:
    #     landmark_list = [[0] for _ in range(n_divide + 1)]
    def m1(arr):
        landmark_list = [tf.reduce_min(arr) for _ in range(n_divide + 1)]
        return landmark_list
    def m2(arr):
        hist = tf.histogram_fixed_width(arr, value_range = [0,tf.reduce_max(arr)], nbins = tf.cast(tf.reduce_max(arr),tf.int32))
        arr_mean = tf.cast(tf.reduce_mean(arr),tf.int32)
        if tf.errors.InvalidArgumentError:
            black_peak = tf.cast(0, tf.int32)
            white_peak = arr_mean + tf.cast(tf.argmax(hist[arr_mean:]), tf.int32)
        else :
            print('arr_mean : ', arr_mean)
            print('hist[:arr_mean] : ', hist[:arr_mean])
            black_peak = tf.cast(tf.argmax(hist[:arr_mean]),tf.int32)
            white_peak = arr_mean + tf.cast(tf.argmax(hist[arr_mean:]),tf.int32)
        valley = hist[black_peak:white_peak]
        # consider only points over 10
        # over_cutline = tf.where(valley > 10)
        # print('over_cutline : ', sess.run(over_cutline))
        # find local minmums among 500 points
        # local_mins = argrelextrema(valley[over_cutline], np.less, order=250)
        # take first local minimum
        # local_min = over_cutline[0][local_mins[0][0]]
        # local_min = argrelextrema(valley, np.less, order=250)[0][np.where(arr_hist_sd[argrelextrema(valley, np.less, order=250)] > 10)[0][0]]
        black_peak_val = tf.log(tf.cast(hist[black_peak],tf.float32))
        white_peak_val = tf.log(tf.cast(hist[white_peak],tf.float32))
        fuzzy_log = ((tf.cast(black_peak,tf.float32) * black_peak_val) + (tf.cast(white_peak,tf.float32) * white_peak_val)) / (black_peak_val + white_peak_val)


        threshold_dict = {}     # 'fuzzy', 'mean', 'median', 'valley', 'fuzzy_log'
        threshold_dict['fuzzy_log'] = fuzzy_log
        threshold_dict['fuzzy'] = tf.cast(((black_peak + white_peak) // 2),tf.int32)
        threshold_dict['mean'] = arr_mean
        threshold_dict['median'] = tf.contrib.distributions.percentile(arr, 50.0)
        # threshold_dict['valley'] = black_peak + local_min

        pc1 = tf.cast(threshold_dict[threshold], tf.float32)
        pc2 = tf.contrib.distributions.percentile(arr, max_percent)
        pc1 = tf.cond(tf.cast(pc1,tf.float32) > pc2, lambda: tf.contrib.distributions.percentile(arr, max_percent - 20),lambda: pc1)
        ioi = tf.gather(arr, tf.where(tf.logical_and(arr >= pc1,arr <= pc2)))
        landmark_list = [tf.contrib.distributions.percentile(ioi, i * (100 / n_divide)) for i in range(n_divide) if not i == 0]
        landmark_list = [pc1] + landmark_list + [pc2]

        if standard:
            std_scale = (scale / pc2)
            landmark_list = [landmark * std_scale for landmark in landmark_list]
        return landmark_list
    landmark_list = tf.cond(tf.equal(tf.cast(tf.reduce_max(arr), tf.int32),0), lambda: m1(arr), lambda: m2(arr))

    return [tf.cast(landmark, tf.int32) for landmark in landmark_list]

def rescale_tensor(arr, in_min,in_max, out_min,out_max):

    scaled_arr = (arr - tf.cast(in_min,tf.float32)) / tf.cast((in_max - in_min),tf.float32)
    # print('middle : ', sess.run(tf.reduce_min(scaled_arr)), sess.run(tf.reduce_max(scaled_arr)))
    scaled_arr = tf.cast(scaled_arr,tf.float32) * tf.cast((out_max - out_min),tf.float32) + tf.cast(out_min,tf.float32)

    return scaled_arr

def hm_rescale_tensor(arr, input_landmark_list, standard_landmark_list,first=True):
    # arr_shape = arr.shape
    arr_shape = tf.shape(arr)
    if arr.get_shape().ndims > 1:
        arr = tf.reshape(arr, [-1])
    arr_copy = tf.zeros_like(arr)

    scale_idx = tf.where(tf.cast(arr,tf.int32) < input_landmark_list[0])
    # 0 ~ pc1 rescale
    scaled_arr = rescale_tensor(tf.gather(arr, scale_idx), tf.cast(tf.reduce_min(tf.gather(arr, scale_idx)),tf.int32), input_landmark_list[0],
                                standard_landmark_list[0]-1, standard_landmark_list[0])
    # scaled_arr = tf.reshape(scaled_arr, [-1])
    with tf.variable_scope('rescale', reuse=True, dtype=tf.float32):
        if first :
            arr_copy_var = tf.Variable(arr_copy, validate_shape=False,name='arr_copy',trainable=False)
            tf.add_to_collection("arr_copy_var", arr_copy_var)
        if first == False:
            arr_copy_var = tf.get_collection('arr_copy_var')[0]

    update = tf.scatter_nd_update(arr_copy_var,
                                  scale_idx,
                                  tf.reshape(scaled_arr, [-1]))
    # pc1 ~ m25 ~ m50 ~ m75 ~ pc2 rescale
    for idx in range(len(input_landmark_list) - 1):
        scale_idx = tf.where(tf.logical_and(tf.cast(arr,tf.int32) >= input_landmark_list[idx], tf.cast(arr,tf.int32) < input_landmark_list[idx+1]))
        scaled_arr = rescale_tensor(tf.gather(arr, scale_idx), input_landmark_list[idx], input_landmark_list[idx+1],standard_landmark_list[idx], standard_landmark_list[idx+1])
        # scaled_arr = tf.reshape(scaled_arr, [-1])
        update = tf.scatter_nd_update(update,
                                      scale_idx,
                                      tf.reshape(scaled_arr, [-1]))
    # pc2 ~ max rescale
    scale_idx = tf.where((tf.cast(arr,tf.int32) >= input_landmark_list[-1]))
    scaled_arr = rescale_tensor(tf.gather(arr, scale_idx), input_landmark_list[-1], tf.cast(tf.reduce_max(tf.gather(arr, scale_idx)),tf.int32),
                                standard_landmark_list[-1], standard_landmark_list[-1] + 1)
    # scaled_arr = tf.reshape(scaled_arr, [-1])

    # print(zeros)
    update = tf.scatter_nd_update(update,
                                  scale_idx,
                                  tf.reshape(scaled_arr, [-1]))

    update = tf.clip_by_value(update, tf.cast(standard_landmark_list[0], tf.float32),
                              tf.cast(standard_landmark_list[-1], tf.float32))
    # tf.cond(tf.size(test_copy) > tf.size(test), lambda: f1(update,arr_copy),lambda : print('need to set first = True'))
    def f1(update,arr_copy):
        update = update[:tf.size(arr_copy)]
        return update
    def f2(update):
        return update
    update = tf.cond(tf.greater(tf.size(update), tf.size(arr_copy)), lambda: f1(update, arr_copy), lambda: f2(update))
    update = update[:tf.size(arr_copy)]
    update = tf.reshape(update,arr_shape)
    print('rescale update : ', update)
    return update

def batch_histogram_match_tensor(input, batch_size, threshold = 'fuzzy', n_divide = 4, standard=False, scale=1,first=True):
    # output = tf.zeros_like(input)
    # print('output shape : ', output.get_shape())
    landmarks_list = []
    print('>>> get landmark')
    for i in range(batch_size):
        landmark_list = cal_hm_landmark_tensor(input[i],threshold=threshold, n_divide = n_divide, standard=standard, scale=scale)
        landmarks_list.append(landmark_list)
    stadard_list = tf.reduce_mean(landmarks_list, axis=0)
    # update = tf.Variable(output,validate_shape=False)
    # print('update shape : ', tf.shape(update))
    print('>>> start rescale')
    batch_match = hm_rescale_tensor(input[0], landmarks_list[0], stadard_list,first)
    zeros = tf.zeros_like([batch_match])
    zeros = tf.concat([zeros, [batch_match]], 0)
    for idx in range(1,batch_size):
        batch_match = hm_rescale_tensor(input[idx], landmarks_list[idx],stadard_list,first)
        zeros = tf.concat([zeros, [batch_match]], 0)
        # batch_match = tf.expand_dims(batch_match, 0)
        # update = tf.scatter_nd_update(update,
        #                               [idx],
        #                               batch_match)

    update = zeros[1:]
    return update

def batch_histogram_match_tensor_undefined(input, threshold = 'fuzzy', n_divide = 4, standard=False, scale=1,first=True):

    landmarks_list = []
    print('>>> get landmark')
    batch_size = tf.shape(input)[0]
    i = tf.constant(0)
    def _cond(i, batches):
        return tf.less(i, tf.cast(batch_size, tf.int32))

    def _body(i, batches):
        landmark_list = cal_hm_landmark_tensor(batches[i], threshold=threshold, n_divide=n_divide, standard=standard,scale=scale)
        landmarks_list.append(landmark_list)
        i = tf.add(i, 1)
        return landmarks_list

    landmarks_list = tf.while_loop(_cond, _body, [i, input])

    stadard_list = tf.reduce_mean(landmarks_list, axis=0)
    # update = tf.Variable(output,validate_shape=False)
    # print('update shape : ', tf.shape(update))
    print('>>> start rescale')
    batch_match = hm_rescale_tensor(input[0], landmarks_list[0], stadard_list,first)
    zeros = [batch_match]
    # zeros = tf.concat([zeros, [batch_match]], 0)
    idx = tf.constant(0)
    def _cond1(idx, batches):
        return tf.less(idx, tf.cast(batch_size, tf.int32))

    def _body2(idx, batches):
        batch_match = hm_rescale_tensor(batches[idx], landmarks_list[idx], stadard_list, first)
        # zeros = tf.concat([zeros, [batch_match]], 0)
        zeros = tf.concat([[batch_match], [batch_match]], 0)
        idx = tf.add(idx, 1)
        return idx, zeros

    _, update = tf.while_loop(_cond1, _body2, [idx, input],
                              shape_invariants=[idx.get_shape(), tf.TensorShape([None, None,None,None])])

        # batch_match = tf.expand_dims(batch_match, 0)
        # update = tf.scatter_nd_update(update,
        #                               [idx],
        #                               batch_match)

    # update = zeros[1:]
    return update

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

    print(rpn_match.get_shape())
    print(rpn_class_logits.get_shape())

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

    print(anchor_class.get_shape())
    print(rpn_class_logits.get_shape())

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
    print(rpn_bbox)
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




