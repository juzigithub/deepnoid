
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


example_path = 'D:\\dataset\\cyst\\data2\\00488_X_876894.jpg'

example_img = cv2.imread(example_path, 0)

img = cv2.resize(example_img,(800,400))

## 히스토그램 평활화
# img2 = cv2.equalizeHist(example_img)
# img2 = cv2.resize(img2,(800,400))

## CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img3 = clahe.apply(example_img)

# ## 영상 히스토그램 분석
hist1 = cv2.calcHist([img],[0],None,[256],[0,256])
# # hist2 = cv2.calcHist([img3],[0],None,[256],[0,256])
# #
# plt.subplot(221),plt.imshow(img,'gray'),plt.title('Red Line')
# # plt.subplot(222),plt.imshow(img3,'gray'),plt.title('Green Line')
# plt.subplot(223),plt.plot(hist1,color='r')#,plt.plot(hist2,color='g')
# plt.xlim([0,256])
# plt.show()
print(hist1)
print(hist1.shape)
print(np.ndarray.tolist(hist1))

hist_list = np.ndarray.tolist(hist1)

hist_list[np.argmax(hist_list)] = [0]
thn = np.argmax(hist_list)

#

## Gaussian Blur
blurred = cv2.GaussianBlur(img3, (5, 5), 0)
blurred2 = cv2.GaussianBlur(img, (5, 5), 0)

## otsu
_, th1 = cv2.threshold(img3, 180, 255, cv2.THRESH_BINARY)
_, th2 = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
_, th3 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

## 원본에서 바로 threshold 하는 것 보다 CLAHE 후 gaussian 블러 처리 한 뒤 threshold 하는 것이 더 좋다
_, th11 = cv2.threshold(img, thn, 255, cv2.THRESH_BINARY)
cv2.imshow('_th1.jpg', th1)
cv2.imshow('_th2.jpg', th2)
cv2.imshow('_th3.jpg', th3)
cv2.imshow('_th11.jpg', th11)
cv2.waitKey(0)