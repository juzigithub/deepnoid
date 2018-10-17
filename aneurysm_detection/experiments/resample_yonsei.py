import numpy as np # linear algebra
import pydicom as dicom
import os
import scipy.ndimage
import cv2
import nibabel
import glob

# from skimage import measure, morphology
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection


INPUT_PATH = 'D:\\yonsei\\input\\SEVSH_BA_000002'
LABEL_PATH = 'D:\\yonsei\\label\\SEVSH_BA_000002_LabelData'
IMG_SIZE = [256, 256]

def load_dcm_slices(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def load_nii_slices(path):
    # slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    nii_list = glob.glob(path + '\\*.nii.gz')
    # nii_list = sorted(nii_list)
    slices = [[nibabel.load(n).get_data()] for n in nii_list]
    slices = np.sum(slices, axis=0)[0]
    return slices


def resample(input_slices, label_slices, new_spacing=[0.5,0.5,0.5]):
    image = np.stack([s.pixel_array for s in input_slices]).astype(np.int16)
    print('dcm_ori', image.shape)

    # Determine current pixel spacing
    # print('thick', input_slices[0].SliceThickness)
    # print('pix', input_slices[0].PixelSpacing[0])

    spacing = map(float, ([input_slices[0].SliceThickness, input_slices[0].PixelSpacing[0], input_slices[0].PixelSpacing[1]]))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    # print('real_size_factor', real_resize_factor)
    input = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    label = scipy.ndimage.interpolation.zoom(label_slices, real_resize_factor)

    return input, label, new_spacing

def resize_imgs(imgs):
    img_list = []
    # clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    for img in imgs:
        img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]), interpolation=cv2.INTER_AREA)
        print('imgs_shape', img.shape)
        # img = clahe.apply(img.astype(np.uint8))

        # img = scale(img.flatten())
        # img = np.reshape(img, (IMG_SIZE[0], IMG_SIZE[1]))

        # img = (img - np.mean(img)) / np.max(img) ##################################
        img_list.append(img)
    return np.array(img_list)

dcm_slices = load_dcm_slices(INPUT_PATH)
nii_slices = load_nii_slices(LABEL_PATH).astype(np.uint8)
nii_slices = np.transpose(nii_slices, (2, 1, 0))
print('nii_ori', nii_slices.shape)
resampled_dcm, resampled_nii, new_spacing = resample(dcm_slices, nii_slices)

print('dcm_after',resampled_dcm.shape)
print('nii_after',resampled_nii.shape)
print(new_spacing)


resampled_dcm = resize_imgs(resampled_dcm)
print('max', np.max(resampled_dcm), 'min', np.min(resampled_dcm))

np.savez_compressed(os.path.join(INPUT_PATH, 'input_resampled.npz'), all=resampled_dcm)
np.savez_compressed(os.path.join(LABEL_PATH, 'label_resampled.npz'), all=resampled_nii)


# for idx, d, n in zip(range(len(resampled_dcm)), resampled_dcm, resampled_nii):
#     cv2.imwrite('d:\\a\\{}.png'.format(idx), d * 255)
#     cv2.imwrite('d:\\b\\{}.png'.format(idx), n * 255)