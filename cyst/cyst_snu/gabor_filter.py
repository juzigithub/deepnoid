from skimage.segmentation import active_contour
import matplotlib.pyplot as plt
from skimage import morphology
import cv2
import json
import numpy as np
import phasepack

# path
json_path = 'D:\\dataset\\cyst\\ms\\00171_X_426937_he.json'
img_path = json_path.replace('.json', '.jpg')
upper_label_path = img_path.replace('.jpg', '_u_label.jpg')
lower_label_path = img_path.replace('.jpg', '_l_label.jpg')


def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = 0.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[ npzarray > 1 ] = 1.
    npzarray[ npzarray < 0 ] = 0.
    return npzarray

def make_label(json_path, img_path, upper_label_path, lower_label_path):
    ## Labelme JSON FILE 구조
    ## JSON -> shapes[class_n] -> 좌표점 : points, 클래스명 : label
    origin_img = cv2.imread(img_path)

    with open(json_path) as data_file:
       data = json.load(data_file)

    shapes = data['shapes']
    upper_pts = np.array(shapes[0]['points'])    # [[ 563.  321.]  [ 890.  295.] [1180.  294.] ...]
    lower_pts = np.array(shapes[1]['points'])

    h, w, _ = origin_img.shape

    for shape in shapes:
       if shape['label'] == 'upper_jaw':
           u_l_img = cv2.imread(upper_label_path)

           if u_l_img is None:
               u_l_img = np.zeros((h, w))
               upper_label = cv2.fillPoly(u_l_img, np.int32([upper_pts]), (255, 255, 255))
               cv2.imwrite(upper_label_path, upper_label)

       elif shape['label'] == 'lower_jaw':
           l_l_img = cv2.imread(lower_label_path)

           if l_l_img is None:
               l_l_img = np.zeros((h, w))
               lower_label = cv2.fillPoly(l_l_img, np.int32([lower_pts]), (255, 255, 255))
               cv2.imwrite(lower_label_path, lower_label)


def build_filters(img_size, show_filters=False):
    # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
    # ksize - size of gabor filter (n, n)
    # sigma - standard deviation of the gaussian function
    # theta - orientation of the normal to the parallel stripes
    # lambda - wavelength of the sunusoidal factor
    # gamma - spatial aspect ratio
    # psi - phase offset
    # ktype - type and range of values that each pixel in the gabor kernel can hold
    # for i in range(4):

    filters = []
    for theta in np.arange(np.pi / 8, np.pi, np.pi / 4):
        kern = cv2.getGaborKernel(ksize = (img_size[1]//20, img_size[0]//20),
                                  sigma = 2.0,
                                  theta = theta,
                                  lambd = 6.0,
                                  gamma = 0.5,
                                  psi = 0,
                                  ktype=cv2.CV_32F)
        if show_filters :
            cv2.imshow(str(theta), kern)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


if __name__ == '__main__':
    make_label(json_path, img_path, upper_label_path, lower_label_path)
    ### load upper label ###
    u_label = cv2.imread(upper_label_path)
    u_label = cv2.cvtColor(u_label, cv2.COLOR_BGR2GRAY)

    ### load lower label ###
    l_label = cv2.imread(lower_label_path)
    l_label = cv2.cvtColor(l_label, cv2.COLOR_BGR2GRAY)

    ### load img ###
    origin_img = cv2.imread(img_path)
    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)

    ### pre-processing ###

    # img = cv2.equalizeHist(origin_img)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    img = morphology.erosion(origin_img, np.ones([2, 2]))
    img = morphology.dilation(img, np.ones([3, 3]))

    ### gabor-filtering ###
    filters = build_filters(np.shape(img)[:2], show_filters=False)
    gabor = process(img, filters)

##################################################################################
    # cv2.imshow('gabor', gabor)
    a = (img - gabor)
    # a = 255 - a
    cv2.imshow('a',a)
    a = a * l_label
    cv2.imshow('l_label', l_label)
    cv2.imshow('aa',a)

    _, th = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # th = morphology.dilation(th, np.ones([3,3]))

    cv2.imshow('th',th)

    a = a * th
    # a = morphology.erosion(a, np.ones([1,1]))
    # cv2.imshow('aaa',a)
    cv2.imshow('aaa',cv2.resize(a, (1200,600)))

##################################################################################
    ### false-positive elmination ###
    u_fp = gabor * u_label
    l_fp = gabor * l_label

    ### otsu thresholding ###
    # false-positive elmination 없이 바로 threshold
    _, gabor_threshold = cv2.threshold(gabor, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # false-positive elmination 후 threshold
    _, u_fp_threshold = cv2.threshold(u_fp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, l_fp_threshold = cv2.threshold(l_fp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



    ### masking ###
    u_gabor_threshold_fp = gabor_threshold * u_label * img
    u_gabor_fp_threshold = u_fp_threshold * img
    l_gabor_threshold_fp = gabor_threshold * l_label * img
    l_gabor_fp_threshold = l_fp_threshold * img

    ### canny edge detector ###
    # canny_img
    canny = cv2.Canny(img, 80, 160)
    canny = morphology.dilation(canny, np.ones([3,3]))
    # canny_gabor
    b = cv2.Canny(gabor, 50, 100)


    ### phasecongmono ###
    # m, ori, ft, _ = phasepack.phasecongmono(img)
    # canny = ft


    ### canny * gabor ###
    c = a * b
    ################



    ### show img ###
    cv2.imshow('1. origin_img', origin_img)
    cv2.imshow('2. after pre-processing', img)
    cv2.imshow('3. gabor_filtered', gabor)
    cv2.imshow('4. false-positive elimination', l_fp)
    cv2.imshow('5. otsu_thresholding', l_fp_threshold)
    cv2.imshow('6. l_gabor_fp_threshold', l_gabor_fp_threshold)
    cv2.imshow('7. l_gabor_threshold_fp', l_gabor_threshold_fp)
    cv2.imshow('8. gabor_threshold', gabor_threshold)
    cv2.imshow('9. canny', canny)
    cv2.imshow('final', cv2.resize(c, (1200,600)) )

    ### Phasecongmono return ###
    # M       Maximum moment of phase congruency covariance, which can be used
    #         as a measure of edge strength
    # ori     Orientation image, in integer degrees (0-180), positive angles
    #         anti-clockwise.
    # ft      Local weighted mean phase angle at every point in the image. A
    #         value of pi/2 corresponds to a bright line, 0 to a step and -pi/2
    #         to a dark line.
    # T       Calculated noise threshold (can be useful for diagnosing noise
    #         characteristics of images). Once you know this you can then specify
    #         fixed thresholds and save some computation time.
    # cv2.imshow('a', np.array(m))
    # cv2.imshow('b', np.array(ori))
    # cv2.imshow('c', np.array(ft))
    # data = np.multiply(a, canny)
    # xdata = data[:, 0]
    # ydata = data[:, 1]
    # print(data[0,:])
    # print(np.shape(data))
    # z = np.polyfit(xdata, ydata, 5)
    # f = np.poly1d(z)
    #
    # print(np.shape(f))
    # print(f)
    # # t = np.arange(0, edges.shape[1], 1)
    # # ax2.plot(t, f(t))


    cv2.waitKey(0)
    cv2.destroyAllWindows()











    # def active_contour(image, snake, alpha=0.01, beta=0.1,
    #                    w_line=0, w_edge=1, gamma=0.01,
    #                    bc='periodic', max_px_move=1.0,
    #                    max_iterations=2500, convergence=0.1):
    # canny = active_contour(canny, init, bc='fixed',alpha=0.015, beta=10, gamma= 0.001)

    # cv2.imshow('init', init)
    # print(width, height)

    # def phasecongmono(img, nscale=5, minWaveLength=3, mult=2.1, sigmaOnf=0.55,
    #                   k=2., cutOff=0.5, g=10., noiseMethod=-1, deviationGain=1.5):
    ### active_contour ###
    # s = np.linspace(0, 2 * np.pi, 100)
    # x = width//2 + width//2 * np.cos(s)
    # y = height//2 + height//2 * np.sin(s)
    # init = np.array([x, y]).T
    # init = np.array([x, y])


    # init = cv2.imread('D:\\dataset\\cyst\\data2\\label.jpg')
    # init = cv2.cvtColor(init, cv2.COLOR_BGR2GRAY)
    # print(np.shape(init))
    # init = 255 - init
    # canny = active_contour(canny, init)
    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.imshow(img)
    # ax.plot(canny[:,0], canny[:,1], '-b', lw=3)
    # ax.set_xticks([]), ax.set_yticks([])
    # plt.show()
    # a = PhaseCoherence(50., otsu_res_filtered ,1000)
    # print(np.shape(a))





    # cv2.imshow('d', u_label)
    # cv2.imshow('da', l_label)
    # cv2.imshow('canny', canny)
    # cv2.imshow('original', img)
    # cv2.imshow('result', res1)
    # cv2.imshow('img * otsu', img * th3)
    # cv2.imshow('upper_roi', img * th3 * u_label)
    # cv2.imshow('lower_roi', img * th3 * l_label)
    # cv2.imshow('img * otsu', img * (th3 + canny))
    # cv2.imshow('plus', canny * res1)

