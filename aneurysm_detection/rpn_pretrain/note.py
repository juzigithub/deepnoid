import numpy as np

#
# input_path = 'd:\\input_data.npy'
# label_path = 'd:\\label_data.npy'
# label_path = 'd:\\rpn_pretrain_label_ori_256.npz'
#
# input = np.load(input_path)
# label = np.load(label_path)['all']
#
# print(input.shape)
# print(label[100])
# print(label.shape)
# print(input[0])
# print(label)
# print(np.round(label[1:] * 256))


# a = np.arange(0,9)
# a = a.reshape((3,3))
# print(a)
#
# a = []
# b = np.loadtxt('d:\\FILE00084.txt', delimiter=' ')
# # print(b)
# c = np.loadtxt('d:\\FILE00085.txt', delimiter=' ')
# # print(c)
# a.append(b)
# a.append(c)
#
# np.savez_compressed('d:\\a.npz', all=a)



aa = np.load('d:\\a.npz')['all']
# print(aa[0].shape)
# print(aa[1].shape)
print(np.ndim(aa[0]))
print(np.ndim(aa[1]))
#
for a in aa :
    if np.ndim(a) == 1:
        a = np.expand_dims(a, 0)
    a[:, 1:] = np.round(a[:, 1:] * 256)

    print('bbox',a[:,1:])
    print('class',a[:,0])

    rpn_class_label = np.expand_dims(a[:, 0], -1).reshape((1, -1, 1))
    print('class_label',rpn_class_label.shape)
    rpn_bbox_label = a[:, 1:].reshape((1, -1, 4))
    print('bbox_label', rpn_bbox_label.shape)
#




# RPN_ANCHOR_SCALES = [int(256 / i) for i in [32, 16, 8, 4, 2]]
# BACKBONE_STRIDES = [4, 8, 16, 32, 64]

# print(RPN_ANCHOR_SCALES)
# print(BACKBONE_STRIDES)