import numpy as np

# a = [[[1,2,3,4],
#       [0,1,2,3]],
#      [[1,2,3,4],
#       [3,2,1,0]]]
#
# # print(np.add(a[0][:][0],1))
# # print(np.subtract(a[0][:][1],1))
# # print(np.multiply(a[0][:][2],2))
# # print(np.divide(a[0][:][3],3))
# a = np.array(a, dtype=np.float32)
# print(a[0])
# a[0,:,0] = (a[0,:,0] - np.mean(a[0,:,0]))/2
# print(a[0,:,1])
# print(a[0,:,2])
# print(a[0,:,3])
#
# print(a)

# a = np.array([[[[1,2,3],
#                 [4,5,6]],
#                [[10,11,12],
#                 [13,14,15]]],
#
#               [[[16,17,18],
#                 [19,20,21]],
#                [[22,23,24],
#                 [25,26,27]]],
#
#               [[[28,29,30],
#                 [31,32,33]],
#                [[34,35,36],
#                 [37,38,39]]]])
#
# print(a)
# print(np.shape(a))
# b = np.transpose(a, [0, 3, 1, 2])
# print(b)
# print(np.shape(b))
#
# c = np.reshape(b, [-1, 2, 2])
# print(c)
# print(np.shape(c))


# import nibabel
# import cv2
# a = nibabel.load('D:\\dataset\\someones_epi.nii.gz').get_fdata()
# print(np.shape(a))
# b = np.max(a[26,:,:])
# print(b)
# cv2.imshow('a', a[26,:,:]/b)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# a = 1
# for i in [a]:
#     print(1)


path = 'C:\\Users\\sunki\\PycharmProjects\\deepnoid\\brats2018\\'
# seg = np.load(path + 'brats_label_chunk_2.npy')
# # print(seg[0][0][0])
# a = []
# for n in range(300):
#     for i in range(240):
#         for j in range(240):
#             a.append(seg[n][i][j][0])
#             # print(seg[n][i][j][0])
# # print(a)
# print(type(a))
# b = set(a)
# print(b)



# a = np.concatenate([np.load(path + 'brats_label_chunk_{}.npy'.format(i)) for i in range(1)], axis=0)
# a = [i for i in range(5)]
# a.remove(0)
# print(np.shape(a))



# for i in range(3):
#     a = np.load(path + 'brats_label_chunk_{}.npy'.format(i))
#     print(np.shape(a))
# a1 = [[1,2]]
# a2 = [[3,4]]
# a3 = [[5,6]]
#
# d = np.concatenate([eval('a{}'.format(i)) for i in range(1,4)], axis=0)
# print(d)

# train_idx = [i for i in range(5)]
# for i in range(3):
#     train_idx.remove(i)
# print(train_idx)

# print(train_idx[1:])

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import directed_hausdorff
def iou_coe(output, target, smooth=1e-5):
    # output : self.foreground_pred
    # target : self.foreground_truth
    # return : list of Batch IoU

    axis = [1,2,3]
    pre = tf.cast(output > 0.51, dtype=tf.float32)
    truth = tf.cast(target > 0.51, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    batch_iou = (inse + smooth) / (union + smooth)
    iou = batch_iou

    return iou, inse, pre, union

with tf.Session() as sess:
    a = np.array([[[0],
                  [4],
                  [5],
                  [7]],
                  [[0],
                  [4],
                  [5],
                  [4]]],
                 dtype=np.float32)
    label = np.array([[[4],
                  [4],
                  [5],
                  [0]],
                  [[0],
                  [4],
                  [7],
                  [0]]],
                 dtype=np.float32)
    sess.run(tf.global_variables_initializer())
    # print(np.shape(a))

    # a[a==4.] =1.
    # a[a==5.] =2.
    # a[a==7.] =3.

    key = np.array([0,1,2,3], dtype=np.float32)

    _, index1 = np.unique(a, return_inverse=True)
    _, index2 = np.unique(label, return_inverse=True)
    print(index1)
    print(index2)
    a = key[index1].reshape(a.shape)
    label = key[index2].reshape(label.shape)


    a = tf.one_hot(a, 4)
    label = tf.one_hot(label, 4)

    # a = tf.argmax(a, axis=3)
    # label = tf.argmax(label, axis=3)
    #
    # ad = tf.diag_part(np.arange(9).reshape(3,3))
    # af = tf.layers.flatten(a)
    # labelf = tf.layers.flatten(label)

    # af = tf.squeeze(a, axis=0)
    # labelf = tf.squeeze(label, axis=0)

    # af = tf.reshape(a, [-1])
    # labelf = tf.reshape(label, [-1])

    # cm = tf.confusion_matrix(labelf, af, num_classes=len([0,1,2,3]))
    # cm = sess.run(cm)
    # print(cm)
    # FP = tf.reduce_sum(cm, axis=0) - tf.diag_part(cm)
    # FN = tf.reduce_sum(cm, axis=1) - tf.diag_part(cm)
    # TP = tf.diag_part(cm)
    # TN = tf.reduce_sum(cm) - (FP + FN + TP)
##############################################
    # a = sess.run(a)
    # af = sess.run(af)
    # labelf = sess.run(labelf)
    # ad = sess.run(ad)
    # print('af', af)
    # print('labelf', labelf)
    # print(np.arange(9).reshape(3,3))
    # print('ad', ad)
    # print(confusion_matrix(labelf[0], af[0]))

    # print(label)
    # print(confusion_matrix(labelf, af))
    # confusion_matrix = confusion_matrix(labelf, af)
##############################################
    import numpy as np
    from numpy.core.umath_tests import inner1d


    # A = np.array([[1,2],[3,4],[5,6],[7,8]])
    # B = np.array([[2,3],[4,5],[6,7],[8,9],[10,11]])

    # Hausdorff Distance
    def HausdorffDist(A, B):
        # Hausdorf Distance: Compute the Hausdorff distance between two point
        # clouds.
        # Let A and B be subsets of metric space (Z,dZ),
        # The Hausdorff distance between A and B, denoted by dH(A,B),
        # is defined by:
        # dH(A,B) = max(h(A,B),h(B,A)),
        # where h(A,B) = max(min(d(a,b))
        # and d(a,b) is a L2 norm
        # dist_H = hausdorff(A,B)
        # A: First point sets (MxN, with M observations in N dimension)
        # B: Second point sets (MxN, with M observations in N dimension)
        # ** A and B may have different number of rows, but must have the same
        # number of columns.
        #
        # Edward DongBo Cui; Stanford University; 06/17/2014

        # Find pairwise distance
        D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
        # Find DH
        dH = np.max(np.array([np.max(np.min(D_mat, axis=0)), np.max(np.min(D_mat, axis=1))]))
        return (dH)




    def cal_result(pred, label, one_hot=False):
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

        cm = confusion_matrix(_label1, _pred1, labels=[0,1])

        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (FP + FN + TP)

        # accuracy, sensitivity, specificity, mean iou, dice coefficient, hausdorff
        acc = (TP + TN).sum() / (TP + FP + FN + TN).sum()
        sens = TP.sum() / (TP + FN).sum()
        spec = TN.sum() / (TN + FP).sum()
        miou = (TP / (FP + FN + TP)).mean()
        dice = (2*TP).sum() / (2*TP + FP + FN).sum()
        hdorff = max(directed_hausdorff(_pred2, _label2)[0], directed_hausdorff(_label2, _pred2)[0])

        return acc, sens, spec, miou, dice, hdorff
##########################################################
    def cal_result_tensor(pred, label, n_class, one_hot=True):
        # convert one-hot labels to multiple labels
        if one_hot:
            _pred = tf.argmax(pred, axis=-1)
            _label = tf.argmax(label, axis=-1)
        else:
            _pred = pred
            _label = label
        __pred = tf.reshape(_pred, [tf.shape(_pred)[0], -1])
        __label = tf.reshape(_label, [tf.shape(_label)[0], -1])

        _pred = tf.reshape(_pred, [-1])
        _label = tf.reshape(_label, [-1])

        # make confusion matrix
        cm = tf.confusion_matrix(labels=_label, predictions=_pred, num_classes=n_class)

        # count TP,FP,FN,TN
        TP = tf.diag_part(cm)
        FP = tf.reduce_sum(cm, axis=0) - TP
        FN = tf.reduce_sum(cm, axis=1) - TP
        TN = tf.reduce_sum(cm) - (FP + FN + TP)

        # cal evaluation
        accuracy = tf.reduce_sum(TP + TN) / tf.reduce_sum(TP + FP + FN + TN)
        sensitivity = tf.reduce_sum(TP) / tf.reduce_sum(TP + FN)
        specificity = tf.reduce_sum(TN) / tf.reduce_sum(TN + FP)
        mean_iou = tf.reduce_mean(TP / (FP + FN + TP))
        dice = tf.reduce_sum(2 * TP) / tf.reduce_sum(2 * TP + FP + FN)

        print(max(directed_hausdorff(sess.run(__pred), sess.run(__label))[0], directed_hausdorff(sess.run(__label), sess.run(__pred))[0]))

        return accuracy, sensitivity, specificity, mean_iou, dice, cm
##################################################
    # acc, sens, spec, iou, dice, cm = cal_result(a, label, 4)
    # print(sess.run(acc), sess.run(sens), sess.run(spec), sess.run(iou), sess.run(dice), sess.run(cm))
    #
    aa = a.eval()
    # aa[aa == 1.0]
    aa = np.argmax(aa, axis=-1)
    print(aa)
    et_key = np.array([0,0,0,1], dtype=np.float32)
    tc_key = np.array([0,1,0,1], dtype=np.float32)
    wt_key = np.array([0,1,1,1], dtype=np.float32)
    _, index = np.unique(aa, return_inverse=True)

    et_arr = et_key[index].reshape(aa.shape)
    tc_arr = tc_key[index].reshape(aa.shape)
    wt_arr = wt_key[index].reshape(aa.shape)

    def convert_to_subregions(pred, label, convert_key, one_hot=True):
        if one_hot:
            pred_arr = np.argmax(pred.eval(), axis=-1)
            label_arr = np.argmax(label.eval(), axis=-1)
        else:
            pred_arr = pred.eval()
            label_arr = label.eval()

        key = np.array(convert_key)
        _, pred_index = np.unique(pred_arr, return_inverse=True)
        _, label_index = np.unique(label_arr, return_inverse=True)

        pred_arr = key[pred_index].reshape(pred_arr.shape)
        label_arr = key[label_index].reshape(label_arr.shape)

        return pred_arr, label_arr


    b, c = convert_to_subregions(a, a, [0,0,0,1], True)




    a = np.array([[[0],
                   [1],
                   [3]],
                  [[1],
                   [2],
                   [0]]])
    b = np.eye(4)[a]
    print(b)
    # print(b.flatten())
    # print(b.reshape(np.shape(b)[0],-1))
    # a = [0,5]
    # b = [1,2]
    # print(np.sum(a,b))
    # key = np.array([0,0,0,3], dtype=np.float32)

    # _, index1 = np.unique(a, return_inverse=True)
    # _, index2 = np.unique(label, return_inverse=True)
    # print(index1)
    # print(index2)
    # a = key[index1].reshape(a.shape)

    # print(HausdorffDist(sess.run(a), sess.run(label)))
    # FP2 = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    # FN2 = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    # TP2 = np.diag(confusion_matrix)
    # TN2 = confusion_matrix.sum() - (FP2 + FN2 + TP2)
    #
    # print(sess.run(TP), sess.run(TN), sess.run(FP), sess.run(FN))
    # print(TP2, TN2, FP2, FN2)

    #
    # ############### Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP / (TP + FN)
    # ############### Specificity or true negative rate
    # TNR = TN / (TN + FP)
    # # Precision or positive predictive value
    # PPV = TP / (TP + FP)
    # # Negative predictive value
    # NPV = TN / (TN + FN)
    # # Fall out or false positive rate
    # FPR = FP / (FP + TN)
    # # False negative rate
    # FNR = FN / (TP + FN)
    # # False discovery rate
    # FDR = FP / (TP + FP)
    #
    # ############### Overall accuracy
    # ACC = (TP + TN) / (TP + FP + FN + TN)
    # ############### Mean iou
    # 클래스마다  TP / (FP + FN + TP) 해주고 평균 내기
    # ############### Dice score
    # DICE = (2 * TP) / (2 * TP + FP + FN)

    # print(a)
    # print(af)
    # label = sess.run(label)
    # print('a',a)
    # print('label',label)
    #
    # iou, inse, _, union = list(iou_coe(a, label))
    #
    # accs = sess.run(inse)
    # iou = sess.run(iou)
    # union = sess.run(union)
    # print(iou)
    # print(accs)
    # print(union)
#################
'''
cfg에 ONE_HOT = True
그 경우 loader 의 seg 는 one_hot 형식으로 저장
acc, sensitivity 측정시 one_hot -> multi label 로 변환 후 배치 마다 reshape(a, [-1]) 후 confusion matrix 통과 
confusion matrix -> acc, sens, iou 등은 utils에 하나의 함수로 만든 후 model.py 의 self.result로 불러오기(one_hot 여부도 변수로)
evaluation 결과 모아서 전체 mean std 구하기 
'''

def binary_cal_result(pred, label, one_hot=False):
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

    cm = confusion_matrix(_label1, _pred1, labels=[0,1])
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[0][0]

    # accuracy, sensitivity, specificity, mean iou, dice coefficient, hausdorff
    acc = (TP + TN) / (TP + FP + FN + TN)
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    miou = TP / (FP + FN + TP)
    dice = (2*TP) / (2*TP + FP + FN)
    hdorff = max(directed_hausdorff(_pred2, _label2)[0], directed_hausdorff(_label2, _pred2)[0])

    return acc, sens, spec, miou, dice, hdorff


a = np.array([[[0],
                  [1],
                  [1],
                  [0]],
                  [[0],
                  [1],
                  [1],
                  [1]]],
                 dtype=np.float32)
label = np.array([[[1],
              [1],
              [1],
              [0]],
              [[0],
              [1],
              [0],
              [1]]],
             dtype=np.float32)
# FN, FP, FN, TN, TN, TP, FP, TP

print(binary_cal_result(label, a))