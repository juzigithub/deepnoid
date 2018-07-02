import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans
from skimage import morphology, measure, io
from matplotlib import pyplot as plt
from scipy import ndimage as ndi




def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = 0.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[ npzarray > 1 ] = 1.
    npzarray[ npzarray < 0 ] = 0.
    return npzarray




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
    img = img - mean
    img = img / std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
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
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img, np.ones([1, 1]))
    dilation = morphology.dilation(eroded, np.ones([6, 6]))


    # dilation = morphology.erosion(thresh_img, np.ones([1, 1]))
    # dilation = morphology.dilation(dilation, np.ones([6, 6]))

    labels = measure.label(dilation)  # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 5 and B[
            2] < col_size / 5 * 4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation

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


if __name__ == '__main__':
    IMG_PATH = 'D:\\dataset\\cyst\\data2\\'

    # img, origin, spacing = load_itk_image(IMG_PATH + '00014_X_029692.jpg')
    itkimage = io.imread(IMG_PATH + '00488_X_876894.jpg', as_grey=True)

    # img = np.reshape(itkimage, [np.shape(itkimage)[0], np.shape(itkimage)[1], 1])
    img = itkimage
    print(np.shape(img))
    img = normalizePlanes(img)
    # img = np.transpose(img, [2,1,0])
    # print(np.shape(img))
    make_lungmask(img, True)