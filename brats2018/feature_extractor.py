from skimage.feature import greycomatrix, greycoprops
import numpy as np                     # numeric python lib
import nibabel
import tensorlayer as tl
import csv
import os
from radiomics import featureextractor
import config as cfg
import SimpleITK as sitk
import scipy.ndimage as ndi            # to determine shape centrality
import pandas as pd

# load csv file and set an index as col 0
data = pd.read_csv(cfg.SURVIVAL_CSV_PATH, index_col=0)

# check columns (variables)
print(data.columns)

# check categorical variables' obs number at each class
print(data["ResectionStatus"].value_counts())

# Subset only status GTR
data = data.loc[data['ResectionStatus'] == "GTR"]

# convert numerical variable to categorical
data['survivor_class'] = 'short_survivor'
data.loc[(data['Survival'] >= 300) & (data['Survival'] < 450), 'survivor_class'] = 'mid_survivor'
data.loc[(data['Survival'] >= 450), 'survivor_class'] = 'long_survivor'
print(data["survivor_class"].value_counts())

print(data)

'''
from scipy.misc import imread
from pylab import rcParams
import matplotlib.image as mpimg       # reading images to numpy arrays
import matplotlib.pyplot as plt        # to plot any graph
import matplotlib.patches as mpatches  # to draw a circle at the mean contour
import cv2
# image processing
rcParams['figure.figsize'] = (6, 6)      # setting default size of plots

data_path = 'C:\\Users\\hshin\\Desktop\\MICCAI_BraTS_2018_Data_Training\\HGG\\Brats18_CBICA_ABM_1'
# reading an image file using matplotlib into a numpy array
# good ones: 11, 19, 23, 27, 48, 53, 78, 218
IMG_SAVE_DATA_PATH = 'C:\\Users\\hshin\\'
# img = mpimg.imread(IMG_SAVE_DATA_PATH + 'af.jpg')
img = imread(IMG_SAVE_DATA_PATH + 'af.jpg', mode='RGB')
# img = mpimg.imread(IMG_SAVE_DATA_PATH + 'tc.jpg')
# img = imread(IMG_SAVE_DATA_PATH + 'tc.jpg', mode='RGB')
plt.imshow(img, cmap='Set3')  # show me the leaf
plt.show()

img = rgb2gray(img) # change to grayscale
# plt.imshow(img)  # show me the leaf
# plt.show()
# using image processing module of scipy to find the center of the leaf
cy, cx = ndi.center_of_mass(img)
# # cy = ndi.center_of_mass(img)
print(cx ,cy)

# img = cv2.imread(IMG_SAVE_DATA_PATH + 'tc.jpg')
# ret,thresh = cv2.threshold(img,127,255,0)
# image, contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# image = cv2.drawContours(img, contours, -1, (0,255,0), 3)
# plt.imshow(image, cmap='gray')  # show me the leaf
# plt.show()

plt.imshow(img, cmap='gray')  # show me the leaf
plt.scatter(cx, cy)           # show me its center44
# # plt.scatter(cy)           # show me its center
plt.show()

# scikit-learn imaging contour finding, returns a list of found edges
contours = measure.find_contours(img, .8)

# from which we choose the longest one
contour = max(contours, key=len)

# let us see the contour that we hopefully found
plt.plot(contour[::,1], contour[::,0], linewidth=0.5)  # (I will explain this [::,x] later)
plt.imshow(img, cmap='gray')
plt.show()

'''

# load survival csv data

survival_id_list = []
survival_age_list =[]
survival_survival_list = []
survival_ResectionStatus_list = []

with open(cfg.SURVIVAL_CSV_PATH, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for idx, content in enumerate(reader):
        if content[3] == 'GTR':
            survival_id_list.append(content[0])
            survival_age_list.append(float(content[1]))
            survival_survival_list.append(int(content[2]))
            survival_ResectionStatus_list.append(content[3])

HGG_path_list = tl.files.load_folder_list(path=cfg.HGG_DATA_PATH)
LGG_path_list = tl.files.load_folder_list(path=cfg.HGG_DATA_PATH)

HGG_id_list = [os.path.basename(p) for p in HGG_path_list]
LGG_id_list = [os.path.basename(p) for p in LGG_path_list]

survival_id_from_HGG = []
survival_id_from_LGG = []

# Separate survival id list in HGG and LGG
for i in survival_id_list:
    if i in HGG_id_list:
        survival_id_from_HGG.append(i)
    elif i in LGG_id_list:
        survival_id_from_LGG.append(i)
    else:
        print(i)

print('len(survival_id_list) : ', len(survival_id_list)) # 59

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
    # print('idx_min : ' ,idx_min)
    # print('idx_max : ' ,idx_max)
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
    glcm = greycomatrix(input, [1], [0], 256, normed=True)
    cont = greycoprops(glcm, 'contrast')
    diss = greycoprops(glcm, 'dissimilarity')
    homo = greycoprops(glcm, 'homogeneity')
    eng = greycoprops(glcm, 'energy')
    corr = greycoprops(glcm, 'correlation')
    ASM = greycoprops(glcm, 'ASM')
    return [cont, diss, homo, eng, corr, ASM]

def save_array_as_nifty_volume(data, filename, reference_name = None):
    transposed_data = np.transpose(data, [2,1,0])
    img = sitk.GetImageFromArray(transposed_data)
    if(reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)

def save_array_as_nifty_volume2(data, filename):
    img = nibabel.Nifti1Image(data, affine=np.eye(4))
    nibabel.save(img, filename)

def get_array_as_nifty_volume(data):
    nii = nibabel.Nifti1Image(data, affine=np.eye(4))
    return nii

# saved label npy to nifty
def save_label_nifty(id_list, data_path):
    for a, data in enumerate(id_list):
        path = os.path.join(data_path, id_list[a], id_list[a])
        et, tc, wt = np.load(id_list[a] + '.npy')

        save_array_as_nifty_volume(et, path + '_seg_et.nii.gz')
        save_array_as_nifty_volume(tc, path+ '_seg_tc.nii.gz')
        save_array_as_nifty_volume(wt, path + '_seg_wt.nii.gz')


        et = np.transpose(et, [1, 0, 2])
        tc = np.transpose(tc, [1, 0, 2])
        wt = np.transpose(wt, [1, 0, 2])

        print('et : ', et.shape)
        print('tc : ', tc.shape)
        print('wt : ', wt.shape)
        print('saved at ' + os.path.join(data_path, id_list[a]))

# save_label_nifty(survival_id_list,cfg.HGG_DATA_PATH)

def add_values_to_key(target_list):
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

# returns HGG/PID/PID only if it's in the survival id list
def get_path_list(data_path):
    id_path_list = []
    path_list = tl.files.load_folder_list(data_path)
    for i in range(len(survival_id_list)):
        for p in path_list:
            if os.path.basename(p) == survival_id_list[i]:
                id_path_list += [os.path.join(data_path, os.path.basename(p), os.path.basename(p))]
    return id_path_list

# id_path_list = get_path_list(cfg.HGG_DATA_PATH)
# print('id_path_list : ', id_path_list)
# print('len(id_path_list) : ', len(id_path_list))        # 163

# get all nifti file path
def nii_names(data_path, train):
    file_list = []  # train [ [flair_path, t1_path, t1ce_path, t2_path, seg_path], ... ]
                    # validation or test [ [flair_path, t1_path, t1ce_path, t2_path],  ... ]
    path_list = get_path_list(data_path)
    for path in path_list:
        flair_path = path + '_flair.nii.gz'
        t1_path = flair_path.replace('flair', 't1')
        t1ce_path = flair_path.replace('flair', 't1ce')
        t2_path = flair_path.replace('flair', 't2')
        seg_path = flair_path.replace('flair', 'seg')
        if train :
            file_list.append([flair_path, t1_path, t1ce_path, t2_path, seg_path] )
        else :
            file_list.append([flair_path, t1_path, t1ce_path, t2_path])
    return np.array(file_list)

file_list = nii_names(cfg.HGG_DATA_PATH,train=False)

def nii_names_for_radiomics(data_path, train):
    file_list = []  # train [ [flair_path, t1_path, t1ce_path, t2_path, seg_path], ... ]
                    # validation or test [ [flair_path, t1_path, t1ce_path, t2_path],  ... ]

    path_list = get_path_list(data_path)

    for path in path_list:
        flair_path = path + '_flair.nii.gz'
        t1_path = flair_path.replace('flair', 't1')
        t1ce_path = flair_path.replace('flair', 't1ce')
        t2_path = flair_path.replace('flair', 't2')
        seg_et_path = flair_path.replace('flair', 'seg_et')
        seg_tc_path = flair_path.replace('flair', 'seg_tc')
        seg_wt_path = flair_path.replace('flair', 'seg_wt')


        if train :
            file_list.append([flair_path, t1_path, t1ce_path, t2_path, seg_et_path, seg_tc_path, seg_wt_path] )
        else :
            file_list.append([flair_path, t1_path, t1ce_path, t2_path])

    # print(np.array(file_list))
    return np.array(file_list)

file_list_rd = nii_names_for_radiomics(cfg.HGG_DATA_PATH,train=True)

def get_radiomics(data_sets):
    total_list = [[] for _ in range(np.shape(data_sets)[-1])]  # [ [flair], [t1], [t1ce], [t2], [seg] ]
    flair_et , t1_et, t1ce_et, t2_et = [], [], [], []
    flair_tc , t1_tc, t1ce_tc, t2_tc = [], [], [], []
    flair_wt , t1_wt, t1ce_wt, t2_wt = [], [], [], []
    for a, data in enumerate(data_sets):
        et, tc, wt = np.load(survival_id_list[a] + '.npy')
        # transpose if needed
        # et = np.transpose(et, [1, 0, 2])
        print(et.shape)
        # check order
        print(data[1])
        print(survival_id_list[a])
        for idx in range(len(total_list)):
            vol = nibabel.load(data[idx]).get_fdata()
            b_min, b_max = [41, 30, 3] , [200, 221, 152]
            # dropped first 2, last 3 scans, and cropped to [160, 192]
            vol = crop_volume_with_bounding_box(vol,b_min,b_max)
            print('vol.shape ; ', vol.shape)        # (160, 192, 150)
            # get sitk image to extract radiomic features
            img = sitk.GetImageFromArray(vol)
            et_label = sitk.GetImageFromArray(et)
            tc_label = sitk.GetImageFromArray(tc)
            wt_label = sitk.GetImageFromArray(wt)
            # extract radiomic fitures using image and label
            extractor = featureextractor.RadiomicsFeaturesExtractor('param.json')
            # it only can accept sitk image or data path
            result_et = extractor.execute(img, et_label)
            result_tc = extractor.execute(img, tc_label)
            result_wt = extractor.execute(img, wt_label)

            if idx == 0:
                flair_et.append(result_et)
                flair_tc.append(result_tc)
                flair_wt.append(result_wt)
            if idx == 1:
                t1_et.append(result_et)
                t1_tc.append(result_tc)
                t1_wt.append(result_wt)
            if idx == 2:
                t1ce_et.append(result_et)
                t1ce_tc.append(result_tc)
                t1ce_wt.append(result_wt)
            if idx == 3:
                t2_et.append(result_et)
                t2_tc.append(result_tc)
                t2_wt.append(result_wt)
    #
    flair_et_dict = add_values_to_key(flair_et)
    flair_tc_dict = add_values_to_key(flair_tc)
    flair_wt_dict = add_values_to_key(flair_wt)

    t1_et_dict = add_values_to_key(t1_et)
    t1_tc_dict = add_values_to_key(t1_tc)
    t1_wt_dict = add_values_to_key(t1_wt)

    t1ce_et_dict = add_values_to_key(t1ce_et)
    t1ce_tc_dict = add_values_to_key(t1ce_tc)
    t1ce_wt_dict = add_values_to_key(t1ce_wt)

    t2_et_dict = add_values_to_key(t2_et)
    t2_tc_dict = add_values_to_key(t2_tc)
    t2_wt_dict = add_values_to_key(t2_wt)

    prefix = ['flair_et_', 'flair_tc_', 'flair_wt_','t1_et_', 't1_tc_', 't1_wt_', 't1ce_et_', 't1ce_tc_', 't1ce_wt_', 't2_et_', 't2_tc_', 't2_wt_']
    all_dict = [flair_et_dict, flair_tc_dict, flair_wt_dict, t1_et_dict, t1_tc_dict, t1_wt_dict, t1ce_et_dict,
                t1ce_tc_dict, t1ce_wt_dict, t2_et_dict, t2_tc_dict, t2_wt_dict]

    # Make dictionaries to one dataframe
    empty = {'id' : survival_id_list}
    empty = pd.DataFrame(empty)
    empty.index = survival_id_list
    for i in range(len(all_dict)):
        new_dict = pd.DataFrame.from_dict(all_dict[i], 'columns')
        new_dict.index = survival_id_list
        new_dict = new_dict.add_prefix(prefix[i])
        empty = pd.concat([empty, new_dict], axis=1)

    # drop general infos
    empty = drop_col_contains_sth(empty, 'general_info')
    print('empty : ', empty)

    # save each dictionaries to csv
    # for i in range(len(all_dict)):
    #     temp = all_dict[i]
    #     with open('dict{}.csv'.format(i), 'w') as f:  # Just use 'w' mode in 3.x
    #         w = csv.DictWriter(f, temp.keys())
    #         w.writeheader()
    #         w.writerow(temp)
    return all_dict, empty

# save radiomic feature dataframe to csv
all_dict, radiomics = get_radiomics(file_list)
radiomics.to_csv('train_radiomic_survival_data.csv')

def get_volumetric(data_sets):

    total_list = [[] for _ in range(np.shape(data_sets)[-1])] # [ [flair], [t1], [t1ce], [t2], [seg] ]

    flair_centroid_x, t1_centroid_x, t1ce_centroid_x, t2_centroid_x = [], [], [], []
    flair_centroid_y, t1_centroid_y, t1ce_centroid_y, t2_centroid_y = [], [], [], []
    flair_centroid_z, t1_centroid_z, t1ce_centroid_z, t2_centroid_z = [], [], [], []
    flair_width, t1_width, t1ce_width, t2_width = [], [], [], []
    flair_length, t1_length, t1ce_length, t2_length = [], [], [], []
    flair_height, t1_height, t1ce_height, t2_height = [], [], [], []
    flair_volume, t1_volume, t1ce_volume, t2_volume = [], [], [], []
    flair_cont, flair_diss, flair_homo, flair_eng, flair_corr, flair_ASM = [], [], [], [], [], []
    t1_cont, t1_diss, t1_homo, t1_eng, t1_corr, t1_ASM = [], [], [], [], [], []
    t1ce_cont, t1ce_diss, t1ce_homo, t1ce_eng, t1ce_corr, t1ce_ASM = [], [], [], [], [], []
    t2_cont, t2_diss, t2_homo, t2_eng, t2_corr, t2_ASM = [], [], [], [], [], []
    et_volume, tc_volume, wt_volume, net_volume, ed_volume = [], [], [], [], []
    et_centroid_x, et_centroid_y, et_centroid_z = [], [], []
    tc_centroid_x, tc_centroid_y, tc_centroid_z = [], [], []
    wt_centroid_x, wt_centroid_y, wt_centroid_z = [], [], []
    et_width, et_length, et_height = [], [], []
    tc_width, tc_length, tc_height = [], [], []
    wt_width, wt_length, wt_height = [], [], []
    flair_cont_et, flair_diss_et, flair_homo_et, flair_eng_et, flair_corr_et, flair_ASM_et = [], [], [], [], [], []
    flair_cont_tc, flair_diss_tc, flair_homo_tc, flair_eng_tc, flair_corr_tc, flair_ASM_tc = [], [], [], [], [], []
    flair_cont_wt, flair_diss_wt, flair_homo_wt, flair_eng_wt, flair_corr_wt, flair_ASM_wt = [], [], [], [], [], []
    t1_cont_et, t1_diss_et, t1_homo_et, t1_eng_et, t1_corr_et, t1_ASM_et = [], [], [], [], [], []
    t1_cont_tc, t1_diss_tc, t1_homo_tc, t1_eng_tc, t1_corr_tc, t1_ASM_tc = [], [], [], [], [], []
    t1_cont_wt, t1_diss_wt, t1_homo_wt, t1_eng_wt, t1_corr_wt, t1_ASM_wt = [], [], [], [], [], []
    t1ce_cont_et, t1ce_diss_et, t1ce_homo_et, t1ce_eng_et, t1ce_corr_et, t1ce_ASM_et = [], [], [], [], [], []
    t1ce_cont_tc, t1ce_diss_tc, t1ce_homo_tc, t1ce_eng_tc, t1ce_corr_tc, t1ce_ASM_tc = [], [], [], [], [], []
    t1ce_cont_wt, t1ce_diss_wt, t1ce_homo_wt, t1ce_eng_wt, t1ce_corr_wt, t1ce_ASM_wt = [], [], [], [], [], []
    t2_cont_et, t2_diss_et, t2_homo_et, t2_eng_et, t2_corr_et, t2_ASM_et = [], [], [], [], [], []
    t2_cont_tc, t2_diss_tc, t2_homo_tc, t2_eng_tc, t2_corr_tc, t2_ASM_tc = [], [], [], [], [], []
    t2_cont_wt, t2_diss_wt, t2_homo_wt, t2_eng_wt, t2_corr_wt, t2_ASM_wt = [], [], [], [], [], []
    flair_brain_wt_ratio, t1_brain_wt_ratio, t1ce_brain_wt_ratio, t2_brain_wt_ratio = [], [], [], []
    flair_brain_tc_ratio, t1_brain_tc_ratio, t1ce_brain_tc_ratio, t2_brain_tc_ratio = [], [], [], []
    flair_brain_et_ratio, t1_brain_et_ratio, t1ce_brain_et_ratio, t2_brain_et_ratio = [], [], [], []
    et_tc_ratio, et_wt_ratio, net_wt_ratio, tc_wt_ratio = [], [], [], []

    for a, data in enumerate(data_sets):
        # check order
        print(data[1])
        print(survival_id_list[a])

        et, tc, wt = np.load(survival_id_list[a] + '.npy')
        # transpose if needed
        # et = np.transpose(et, [1, 0, 2])
        print('et.shape : ', et.shape)

        # get volumes and their ratios
        et_vol = get_volume(et)
        tc_vol = get_volume(tc)
        wt_vol = get_volume(wt)
        net_vol = tc_vol - et_vol
        ed_vol = wt_vol - tc_vol

        et_volume.append(et_vol)
        tc_volume.append(tc_vol)
        wt_volume.append(wt_vol)
        net_volume.append(net_vol)
        ed_volume.append(ed_vol)

        et_tc_ratio1 = et_vol / tc_vol
        et_wt_ratio1 = et_vol / wt_vol
        net_wt_ratio1 = net_vol / wt_vol
        tc_wt_ratio1 = tc_vol / wt_vol

        et_tc_ratio.append(et_tc_ratio1)
        et_wt_ratio.append(et_wt_ratio1)
        net_wt_ratio.append(net_wt_ratio1)
        tc_wt_ratio.append(tc_wt_ratio1)

        # get centroid
        et_cy, et_cx, et_cz = ndi.center_of_mass(et)
        tc_cy, tc_cx, tc_cz = ndi.center_of_mass(tc)
        wt_cy, wt_cx, wt_cz = ndi.center_of_mass(wt)

        et_centroid_x.append(et_cx)
        et_centroid_y.append(et_cy)
        et_centroid_z.append(et_cz)

        tc_centroid_x.append(tc_cx)
        tc_centroid_y.append(tc_cy)
        tc_centroid_z.append(tc_cz)

        wt_centroid_x.append(wt_cx)
        wt_centroid_y.append(wt_cy)
        wt_centroid_z.append(wt_cz)

        # get bounding boxes and it's width, length, height
        _, _, et_w, et_l, et_h = get_ND_bounding_box(et, 0)
        et_min, et_max, _, _, _ = get_ND_bounding_box(et, 3)
        et_width.append(et_w)
        et_length.append(et_l)
        et_height.append(et_h)

        _, _, tc_w, tc_l, tc_h = get_ND_bounding_box(tc, 0)
        tc_min, tc_max, _, _, _ = get_ND_bounding_box(tc, 3)
        tc_width.append(tc_w)
        tc_length.append(tc_l)
        tc_height.append(tc_h)

        _, _, wt_w, wt_l, wt_h = get_ND_bounding_box(wt, 0)
        wt_min, wt_max, _, _, _ = get_ND_bounding_box(wt, 3)
        wt_width.append(wt_w)
        wt_length.append(wt_l)
        wt_height.append(wt_h)

        for idx in range(len(total_list)):
            vol = nibabel.load(data[idx]).get_fdata()
            b_min, b_max = [41, 30, 3] , [200, 221, 152]
            # dropped first 2, last 3 scans, and cropped to [160, 192]
            vol = crop_volume_with_bounding_box(vol,b_min,b_max)
            vol_cropped_et = crop_volume_with_bounding_box(vol, et_min, et_max)
            vol_cropped_tc = crop_volume_with_bounding_box(vol, tc_min, tc_max)
            vol_cropped_wt = crop_volume_with_bounding_box(vol, wt_min, wt_max)

            # get glcm features from image cropped using label locational information
            glcm_per_cropped_volume_et = []
            for z in range(vol_cropped_et.shape[2]):
                img = vol_cropped_et[:,:,z]
                img = np.asarray(img, dtype="uint8")
                result = get_glcm(img)
                glcm_per_cropped_volume_et.append(result)
            et_mean = np.mean(glcm_per_cropped_volume_et, axis = 0)
            et_mean = np.reshape(et_mean, [6,])

            glcm_per_cropped_volume_tc = []
            for z in range(vol_cropped_tc.shape[2]):
                img = vol_cropped_tc[:,:,z]
                img = np.asarray(img, dtype="uint8")
                result = get_glcm(img)
                glcm_per_cropped_volume_tc.append(result)
            tc_mean = np.mean(glcm_per_cropped_volume_tc, axis = 0)
            tc_mean = np.reshape(tc_mean, [6,])

            glcm_per_cropped_volume_wt = []
            for z in range(vol_cropped_wt.shape[2]):
                img = vol_cropped_wt[:,:,z]
                img = np.asarray(img, dtype="uint8")
                result = get_glcm(img)
                glcm_per_cropped_volume_wt.append(result)
            wt_mean = np.mean(glcm_per_cropped_volume_wt, axis = 0)
            wt_mean = np.reshape(wt_mean, [6,])

            # get volume of brain and calculate ration between labels
            volume = get_volume(vol)

            brain_wt_ratio = wt_vol/volume
            brain_tc_ratio = tc_vol/volume
            brain_et_ratio = et_vol/volume

            # glcm of whole image
            glcm_per_volume = []
            for z in range(vol.shape[2]):
                img = vol[:,:,z]
                img = np.asarray(img, dtype="uint8")
                result = get_glcm(img)
                glcm_per_volume.append(result)
            a_mean = np.mean(glcm_per_volume, axis = 0)
            a_mean = np.reshape(a_mean, [6,])

            # centroid of images, width, length, height
            cy, cx, cz = ndi.center_of_mass(vol)
            _, _, w, l, h = get_ND_bounding_box(vol,0)
            # print(' W L H : ', w, l, h)

            if idx == 0 :
                flair_centroid_x.append(cx)
                flair_centroid_y.append(cy)
                flair_centroid_z.append(cz)
                flair_width.append(w)
                flair_length.append(l)
                flair_height.append(h)
                flair_volume.append(volume)
                flair_cont.append(a_mean[0])
                flair_diss.append(a_mean[1])
                flair_homo.append(a_mean[2])
                flair_eng.append(a_mean[3])
                flair_corr.append(a_mean[4])
                flair_ASM.append(a_mean[5])

                flair_cont_et.append(et_mean[0])
                flair_diss_et.append(et_mean[1])
                flair_homo_et.append(et_mean[2])
                flair_eng_et.append(et_mean[3])
                flair_corr_et.append(et_mean[4])
                flair_ASM_et.append(et_mean[5])

                flair_cont_tc.append(tc_mean[0])
                flair_diss_tc.append(tc_mean[1])
                flair_homo_tc.append(tc_mean[2])
                flair_eng_tc.append(tc_mean[3])
                flair_corr_tc.append(tc_mean[4])
                flair_ASM_tc.append(tc_mean[5])

                flair_cont_wt.append(wt_mean[0])
                flair_diss_wt.append(wt_mean[1])
                flair_homo_wt.append(wt_mean[2])
                flair_eng_wt.append(wt_mean[3])
                flair_corr_wt.append(wt_mean[4])
                flair_ASM_wt.append(wt_mean[5])

                flair_brain_wt_ratio.append(brain_wt_ratio)
                flair_brain_tc_ratio.append(brain_tc_ratio)
                flair_brain_et_ratio.append(brain_et_ratio)
            elif idx == 1:
                t1_centroid_x.append(cx)
                t1_centroid_y.append(cy)
                t1_centroid_z.append(cz)
                t1_width.append(w)
                t1_length.append(l)
                t1_height.append(h)
                t1_volume.append(volume)
                t1_cont.append(a_mean[0])
                t1_diss.append(a_mean[1])
                t1_homo.append(a_mean[2])
                t1_eng.append(a_mean[3])
                t1_corr.append(a_mean[4])
                t1_ASM.append(a_mean[5])

                t1_cont_et.append(et_mean[0])
                t1_diss_et.append(et_mean[1])
                t1_homo_et.append(et_mean[2])
                t1_eng_et.append(et_mean[3])
                t1_corr_et.append(et_mean[4])
                t1_ASM_et.append(et_mean[5])

                t1_cont_tc.append(tc_mean[0])
                t1_diss_tc.append(tc_mean[1])
                t1_homo_tc.append(tc_mean[2])
                t1_eng_tc.append(tc_mean[3])
                t1_corr_tc.append(tc_mean[4])
                t1_ASM_tc.append(tc_mean[5])

                t1_cont_wt.append(wt_mean[0])
                t1_diss_wt.append(wt_mean[1])
                t1_homo_wt.append(wt_mean[2])
                t1_eng_wt.append(wt_mean[3])
                t1_corr_wt.append(wt_mean[4])
                t1_ASM_wt.append(wt_mean[5])


                t1_brain_wt_ratio.append(brain_wt_ratio)
                t1_brain_tc_ratio.append(brain_tc_ratio)
                t1_brain_et_ratio.append(brain_et_ratio)
            elif idx == 2:
                t1ce_centroid_x.append(cx)
                t1ce_centroid_y.append(cy)
                t1ce_centroid_z.append(cz)
                t1ce_width.append(w)
                t1ce_length.append(l)
                t1ce_height.append(h)
                t1ce_volume.append(volume)
                t1ce_cont.append(a_mean[0])
                t1ce_diss.append(a_mean[1])
                t1ce_homo.append(a_mean[2])
                t1ce_eng.append(a_mean[3])
                t1ce_corr.append(a_mean[4])
                t1ce_ASM.append(a_mean[5])

                t1ce_cont_et.append(et_mean[0])
                t1ce_diss_et.append(et_mean[1])
                t1ce_homo_et.append(et_mean[2])
                t1ce_eng_et.append(et_mean[3])
                t1ce_corr_et.append(et_mean[4])
                t1ce_ASM_et.append(et_mean[5])

                t1ce_cont_tc.append(tc_mean[0])
                t1ce_diss_tc.append(tc_mean[1])
                t1ce_homo_tc.append(tc_mean[2])
                t1ce_eng_tc.append(tc_mean[3])
                t1ce_corr_tc.append(tc_mean[4])
                t1ce_ASM_tc.append(tc_mean[5])

                t1ce_cont_wt.append(wt_mean[0])
                t1ce_diss_wt.append(wt_mean[1])
                t1ce_homo_wt.append(wt_mean[2])
                t1ce_eng_wt.append(wt_mean[3])
                t1ce_corr_wt.append(wt_mean[4])
                t1ce_ASM_wt.append(wt_mean[5])


                t1ce_brain_wt_ratio.append(brain_wt_ratio)
                t1ce_brain_tc_ratio.append(brain_tc_ratio)
                t1ce_brain_et_ratio.append(brain_et_ratio)
            elif idx == 3:
                t2_centroid_x.append(cx)
                t2_centroid_y.append(cy)
                t2_centroid_z.append(cz)
                t2_width.append(w)
                t2_length.append(l)
                t2_height.append(h)
                t2_volume.append(volume)
                t2_cont.append(a_mean[0])
                t2_diss.append(a_mean[1])
                t2_homo.append(a_mean[2])
                t2_eng.append(a_mean[3])
                t2_corr.append(a_mean[4])
                t2_ASM.append(a_mean[5])

                t2_cont_et.append(et_mean[0])
                t2_diss_et.append(et_mean[1])
                t2_homo_et.append(et_mean[2])
                t2_eng_et.append(et_mean[3])
                t2_corr_et.append(et_mean[4])
                t2_ASM_et.append(et_mean[5])

                t2_cont_tc.append(tc_mean[0])
                t2_diss_tc.append(tc_mean[1])
                t2_homo_tc.append(tc_mean[2])
                t2_eng_tc.append(tc_mean[3])
                t2_corr_tc.append(tc_mean[4])
                t2_ASM_tc.append(tc_mean[5])

                t2_cont_wt.append(wt_mean[0])
                t2_diss_wt.append(wt_mean[1])
                t2_homo_wt.append(wt_mean[2])
                t2_eng_wt.append(wt_mean[3])
                t2_corr_wt.append(wt_mean[4])
                t2_ASM_wt.append(wt_mean[5])

                t2_brain_wt_ratio.append(brain_wt_ratio)
                t2_brain_tc_ratio.append(brain_tc_ratio)
                t2_brain_et_ratio.append(brain_et_ratio)
            else:
                pass

    d = {'flair_centroid_x': flair_centroid_x, 't1_centroid_x': t1_centroid_x, 't1ce_centroid_x': t1ce_centroid_x,
         't2_centroid_x': t2_centroid_x, 'flair_centroid_y': flair_centroid_y, 't1_centroid_y': t1_centroid_y,
         't1ce_centroid_y': t1ce_centroid_y, 't2_centroid_y': t2_centroid_y, 'flair_centroid_z': flair_centroid_z,
         't1_centroid_z': t1_centroid_z ,'t1ce_centroid_z': t1ce_centroid_z, 't2_centroid_z': t2_centroid_z,
         'flair_width': flair_width, 'flair_length': flair_length, 'flair_height': flair_height,
         't1_width': t1_width, 't1_length': t1_length, 't1_height': t1_height,
         't1ce_width': t1ce_width, 't1ce_length': t1ce_length, 't1ce_height': t1ce_height,
         't2_width': t2_width, 't2_length': t2_length, 't2_height': t2_height,
         'flair_volume' : flair_volume, 't1_volume' : t1_volume, 't1ce_volume' : t1ce_volume, 't2_volume': t2_volume,
         'flair_cont' : flair_cont, 't1_cont' : t1_cont, 't1ce_cont' : t1ce_cont, 't2_cont' : t2_cont,
         'flair_diss' : flair_diss, 't1_diss' : t1_diss, 't1ce_diss' : t1ce_diss, 't2_diss' : t2_diss,
         'flair_homo' : flair_homo, 't1_homo' : t1_homo, 't1ce_homo' : t1ce_homo, 't2_homo' : t2_homo,
         'flair_eng' : flair_eng, 't1_eng' : t1_eng, 't1ce_eng' : t1ce_eng, 't2_eng' : t2_eng,
         'flair_corr' : flair_corr, 't1_corr' : t1_corr, 't1ce_corr' : t1ce_corr, 't2_corr' : t2_corr,
         'flair_ASM' : flair_ASM, 't1_ASM' : t1_ASM, 't1ce_ASM' : t1ce_ASM, 't2_ASM' : t2_ASM,
         't1_cont_et' : t1_cont_et, 't1_diss_et' : t1_diss_et, 't1_homo_et' : t1_homo_et,
         't1_eng_et' : t1_eng_et, 't1_corr_et' : t1_corr_et, 't1_ASM_et' : t1_ASM_et,
         't1_cont_tc' : t1_cont_tc, 't1_diss_tc' : t1_diss_tc, 't1_homo_tc' : t1_homo_tc,
         't1_eng_tc' : t1_eng_tc, 't1_corr_tc' : t1_corr_tc, 't1_ASM_tc' : t1_ASM_tc,
         't1_cont_wt' : t1_cont_wt, 't1_diss_wt' : t1_diss_wt, 't1_homo_wt' : t1_homo_wt,
         't1_eng_wt' : t1_eng_wt, 't1_corr_wt' : t1_corr_wt, 't1_ASM_wt' : t1_ASM_wt,
         't1ce_cont_et' : t1ce_cont_et, 't1ce_diss_et' : t1ce_diss_et, 't1ce_homo_et' : t1ce_homo_et,
         't1ce_eng_et' : t1ce_eng_et, 't1ce_corr_et' : t1ce_corr_et, 't1ce_ASM_et' : t1ce_ASM_et,
         't1ce_cont_tc' : t1ce_cont_tc, 't1ce_diss_tc' : t1ce_diss_tc, 't1ce_homo_tc' : t1ce_homo_tc,
         't1ce_eng_tc' : t1ce_eng_tc, 't1ce_corr_tc' : t1ce_corr_tc, 't1ce_ASM_tc' : t1ce_ASM_tc,
         't1ce_cont_wt' : t1ce_cont_wt, 't1ce_diss_wt' : t1ce_diss_wt, 't1ce_homo_wt' : t1ce_homo_wt,
         't1ce_eng_wt' : t1ce_eng_wt, 't1ce_corr_wt' : t1ce_corr_wt, 't1ce_ASM_wt' : t1ce_ASM_wt,
         't2_cont_et' : t2_cont_et, 't2_diss_et' : t2_diss_et, 't2_homo_et' : t2_homo_et,
         't2_eng_et' : t2_eng_et, 't2_corr_et' : t2_corr_et, 't2_ASM_et' : t2_ASM_et,
         't2_cont_tc' : t2_cont_tc, 't2_diss_tc' : t2_diss_tc, 't2_homo_tc' : t2_homo_tc,
         't2_eng_tc' : t2_eng_tc, 't2_corr_tc' : t2_corr_tc, 't2_ASM_tc' : t2_ASM_tc,
         't2_cont_wt' : t2_cont_wt, 't2_diss_wt' : t2_diss_wt, 't2_homo_wt' : t2_homo_wt,
         't2_eng_wt' : t2_eng_wt, 't2_corr_wt' : t2_corr_wt, 't2_ASM_wt' : t2_ASM_wt,
         'flair_cont_et' : flair_cont_et, 'flair_diss_et' : flair_diss_et, 'flair_homo_et' : flair_homo_et,
         'flair_eng_et' : flair_eng_et, 'flair_corr_et' : flair_corr_et, 'flair_ASM_et' : flair_ASM_et,
         'flair_cont_tc' : flair_cont_tc, 'flair_diss_tc' : flair_diss_tc, 'flair_homo_tc' : flair_homo_tc,
         'flair_eng_tc' : flair_eng_tc, 'flair_corr_tc' : flair_corr_tc, 'flair_ASM_tc' : flair_ASM_tc,
         'flair_cont_wt' : flair_cont_wt, 'flair_diss_wt' : flair_diss_wt, 'flair_homo_wt' : flair_homo_wt,
         'flair_eng_wt' : flair_eng_wt, 'flair_corr_wt' : flair_corr_wt, 'flair_ASM_wt' : flair_ASM_wt,
         'et_volume' : et_volume, 'tc_volume' : tc_volume, 'wt_volume' : wt_volume, 'net_volume' : net_volume, 'ed_volume' : ed_volume,
         'et_centroid_x' : et_centroid_x, 'et_centroid_y' : et_centroid_y, 'et_centroid_z' : et_centroid_z,
         'tc_centroid_x' : tc_centroid_x, 'tc_centroid_y' : tc_centroid_y, 'tc_centroid_z' : tc_centroid_z,
         'wt_centroid_x' : wt_centroid_x, 'wt_centroid_y' : wt_centroid_y, 'wt_centroid_z' : wt_centroid_z,
         'et_width' : et_width, 'et_length' : et_length, 'et_height' : et_height,
         'tc_width' : tc_width, 'tc_length' : tc_length, 'tc_height' : tc_height,
         'wt_width' : wt_width, 'wt_length' : wt_length, 'wt_height' : wt_height,
         'et_tc_ratio': et_tc_ratio, 'et_wt_ratio': et_wt_ratio, 'net_wt_ratio': net_wt_ratio, 'tc_wt_ratio': tc_wt_ratio}
    # lengths = {key: len(value) for key, value in d.items()}
    # print(lengths)
    df = pd.DataFrame(data=d)
    print(df)
    return df

df  = get_volumetric(file_list)
df.index = survival_id_list
data = data.join(df)
print(data.columns)
data = data.drop('ResectionStatus', 1)
data = data.drop('Survival', 1)
print(data.columns)

data.to_csv('train_volumetric_survival_data.csv')

radiomics = radiomics.drop('id' , 1)
data = data.join(radiomics)
data.to_csv('train_completed_survival_data.csv')

# data = pd.concat([data, df], axis=1)

print(data)

