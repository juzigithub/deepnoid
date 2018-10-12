import os
import numpy as np
import cv2
import re


def _try_int(ss):
   try:
       return int(ss)
   except:
       return ss


def _number_key(s):
   return [_try_int(ss) for ss in re.split('([0-9]+)', s)]


def _sort_by_number(files):
   files.sort(key=_number_key)
   return files


p = 'D:\\Dataset\\Brain_Aneurysm_Original\\abnorm\\train_img'
p_l = [os.path.join(p, f) for f in os.listdir(p)]
for d in p_l:
   s = _sort_by_number(os.listdir(os.path.join(d, 'img', 'y')))
   # print(s)
   x_list = [os.path.join(d, 'img', 'x', f) for f in s]
   # y_list = [os.path.join(d, 'img', 'y', f) for f in s]

   for x in x_list:
   # for y in y_list:

       m = x.replace('\\Dataset\\Brain_Aneurysm_Original\\abnorm\\train_img\\', '\\Sample\\filtered_input_without_CLAHE\\')
       # m = y.replace('\\Dataset\\Brain_Aneurysm_Original\\abnorm\\train_img\\', '\\Sample\\new_label\\')
       print(m)
       if not os.path.exists(os.path.dirname(m)):
           os.makedirs(os.path.dirname(m))

       input = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
       clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
       input = clahe.apply(input)
       th = np.quantile(a=input, q=0.90)
       _, res = cv2.threshold(input, th, 255, cv2.THRESH_BINARY)
       cv2.imwrite(m, res)