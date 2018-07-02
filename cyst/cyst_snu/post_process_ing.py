IMG_PATH = 'D:\\dataset\\cyst\\data2\\'
maxHU = 1000. # 400
minHU = 0.    # -1000

import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans
from skimage import morphology, measure, io
# from matplotlib import pyplot as plt
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.patches
import cv2

#
#
# def load_itk_image(filename):
#     # itkimage = sitk.ReadImage(filename)
#     itkimage = io.imread(filename, as_grey=True)
#     # print(np.shape(itkimage))
#     itkimage = np.reshape(itkimage, [np.shape(itkimage)[0], np.shape(itkimage)[1], 1])
#     numpyImage = sitk.GetArrayFromImage(itkimage)  # z, y, x axis load
#     numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
#     numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
#
#     return numpyImage, numpyOrigin, numpySpacing

def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = 0.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[ npzarray > 1 ] = 1.
    npzarray[ npzarray < 0 ] = 0.
    return npzarray



def remove_large_objects(ar, max_size=64, connectivity=1, in_place=False):

    if in_place:
        out = ar
    else:
        out = ar.copy()

    if out.dtype == bool:
        selem = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")

    too_large = component_sizes > max_size
    too_large_mask = too_large[ccs]
    out[too_large_mask] = 0

    return out





def make_lungmask(img, display=False):
    """
    # Standardize the pixel value by subtracting the mean and dividing by the standard deviation
    # Identify the proper threshold by creating 2 KMeans clusters comparing centered on soft tissue/bone vs lung/air.
    # Using Erosion and Dilation which has the net effect of removing tiny features like pulmonary vessels or noise
    # Identify each distinct region as separate image labels (think the magic wand in Photoshop)
    # Using bounding boxes for each image label to identify which ones represent lung and which ones represent "every thing else"
    # Create the masks for lung fields.
    # Apply mask onto the original image to erase voxels outside of the lung fields.
    """
    row_size = img.shape[0]
    col_size = img.shape[1]

    mean = np.mean(img)
    std = np.std(img)
    max = np.max(img)
    min = np.min(img)

    # img norm 둘 중 하나만
    # img = (img - mean) / std
    img = (img - min) / (max - min)

    # Find the average pixel value near the lungs
    # to renormalize washed out images
    # middle = img[int(col_size / 3):int(col_size / 3 * 2), int(row_size / 6 * 2 ):int(row_size / 6 * 4)]
    middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]

    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)


    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum
    img[img == max] = mean
    img[img == min] = mean

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())

    # THRESHOLD 넷 중 하나만
    # threshold = np.mean(centers)
    threshold = np.mean(middle)
    # threshold = np.mean(img)

    # w = 3
    # mean = np.mean(img)
    # std = np.std(img)
    # threshold = mean + w * std

    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image


    # non_zero = np.count_nonzero(img == 0.) # 0
    # non_one = np.count_nonzero(img == 1.)  # 1
    # non_zero2 = len(np.where(img ==0.))

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don't want to accidentally clip the lung.

    # erosion 쓰면 작은 치주골 포착 어려움
    dilation = morphology.erosion(thresh_img, np.ones([2, 2]))
    dilation = morphology.dilation(dilation, np.ones([6, 6]))


    # 작은 물체, 큰 물체 없애기
    dilation = remove_large_objects(dilation.astype(bool), 1000)
    dilation = morphology.remove_small_objects(dilation, 150)


    labels = measure.label(dilation.astype(float))  # Different labels are displayed in different colors
    # label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        # if B[2] - B[0] < row_size / 20 and B[3] - B[1] < col_size / 20 and B[0] > row_size / 6 * 3 and B[2] < col_size / 6 * 4:
        if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 5 and B[2] < col_size / 5 * 4:

            good_labels.append(prop.label)
    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0

    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)

    # mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation



#####################################################################################################
    # # create an ellipse
    # el = matplotlib.patches.Ellipse((50, -23), 10, 13.7, 30, facecolor=(1, 0, 0, .2), edgecolor='none')
    #
    # # calculate the x and y points possibly within the ellipse
    # y_int = np.arange(-30, -15)
    # x_int = np.arange(40, 60)
    #
    # # create a list of possible coordinates
    # g = np.meshgrid(x_int, y_int)
    # coords = list(zip(*(c.flat for c in g)))
    #
    # # create the list of valid coordinates (from untransformed)
    # ellipsepoints = np.vstack([p for p in coords if el.contains_point(p, radius=0)])
    #
    # # just to see if this works
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.add_artist(el)
    # ep = np.array(ellipsepoints)
    # ax.plot(ellipsepoints[:, 0], ellipsepoints[:, 1], 'ko')
    # plt.show()
#########################################################################
    # im = cv2.imread("plank.jpg")
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # _, bin = cv2.threshold(thresh_img, 120, 255, 1)  # inverted threshold (light obj on dark bg)
    # bin = cv2.dilate(bin, None)  # fill some holes
    # bin = cv2.dilate(bin, None)
    # bin = cv2.erode(bin, None)  # dilate made our shape larger, revert that
    # bin = cv2.erode(bin, None)
    # # cv2.cvtColor(bin, cv2.COLOR_BGR2GRAY)
    # print(np.shape(bin))
    # bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    # rc = cv2.minAreaRect(contours[0])
    # box = cv2.boxPoints(rc)
    # for p in box:
    #     pt = (p[0], p[1])
    #     print(pt)
    #     cv2.circle(thresh_img, pt, 5, (200, 0, 0), 2)
    # cv2.imshow("plank", thresh_img)
    # cv2.waitKey()

################################################################################






    if display:
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask * img, cmap='gray')
        ax[2, 1].axis('off')
        plt.savefig('t.png')
        plt.show()

    return mask * img
# def chg_VoxelCoord(lists, str, origin, spacing):
#     cand_list = []
#     labels = []
#     # if len(lists) > 2000:
#     for list in lists:
#         if list[0] in str:
#             worldCoord =np.asarray([float(list[3]),float(list[2]),float(list[1])])
#             voxelCoord = worldToVoxelCoord(worldCoord, origin, spacing)
#             if list[4] is '1':
#                 augs, aug_labels = aug_candidate(voxelCoord)
#                 cand_list.append(voxelCoord)
#                 labels.append(int(list[4]))
#                 for aug in augs:
#                     cand_list.append(aug)
#                 al_vec = np.ones((int(aug_labels),1))
#                 for aug_lbl in al_vec:
#                     labels.append(int(aug_lbl))
#             else:
#                 cand_list.append(voxelCoord)
#                 labels.append(int(list[4]))
#     return cand_list, labels

if __name__ == '__main__':
    itkimage = io.imread(IMG_PATH + '00488_X_876894.jpg', as_grey=True)
    img = itkimage
    img = normalizePlanes(img)
    make_lungmask(img, True)