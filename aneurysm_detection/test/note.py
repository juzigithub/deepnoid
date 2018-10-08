import numpy as np
import cv2

img = cv2.imread('d:\\FILE00133.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

label = np.array([[0.33203125, 0.5546875, 0.337890625, 0.55859375]])
label = np.round(label * 255.).astype(np.int32)
print(label)
# cv2.rectangle(img, (x1-5, y1-5), (x2+5, y2+5), (255, 0, 0), 1)

for l in label:
    cv2.rectangle(img, (l[1]-5, l[0]-5), (l[3]+5, l[2]+5), (255, 0, 0), 1)
    cv2.putText(img, '0.51', ( l[1] - 5, l[0] - 7 ), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
    # cv2.putText(img, 'OpenCV', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
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

img = masking_rgb(img, 'red')
cv2.imshow('a', img)
cv2.imwrite('d:\\a.png', img)
cv2.waitKey()

