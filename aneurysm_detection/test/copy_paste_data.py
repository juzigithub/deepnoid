import glob
import os
import shutil
import tensorlayer as tl

path = 'D:\\dataset\\Brain_Aneurysm_new_dataset\\full_data\\newest\\test\\input_dcm\\*\\*\\*\\'
data_list = glob.glob(path + '*.dcm')
# print(data_list)
# print('-----------------------------')
sorted(data_list)
# print(data_list)


before = 'D:\\dataset\\Brain_Aneurysm_new_dataset\\full_data\\dcm\\'
after = 'D:\\dataset\\Brain_Aneurysm_new_dataset\\full_data\\newest\\test\\input_dcm_original\\'

for data in data_list:
    # print('data:', data)
    path = os.path.split(data)[0]
    # print('path:', path)

    path, path2_1 = os.path.split(path)
    # print(path2_1)

    path, path3_1 = os.path.split(path)

    path, path4_1 = os.path.split(path)

    fn_ext = os.path.split(data)[1]
    # print('fn_ext:', fn_ext)
    final_before_path = os.path.join(before, path4_1, fn_ext)
    print('final_before_path:', final_before_path)
    tl.files.exists_or_mkdir(os.path.join(after, path4_1, path3_1, path2_1))
    final_after_path = os.path.join(after, path4_1, path3_1, path2_1, fn_ext)
    print('final_after_path:', final_after_path)
    shutil.copy2(final_before_path, final_after_path)
    print('copied')


