import os

import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
import sqlite3
import glob


root_dir = '/home/hui/database/dataset/zz_other_dataset/hedau/Hedau'
split_path = os.path.join(root_dir, 'traintestind.mat')
img_dir = os.path.join(root_dir, 'Images')
anno_dir = os.path.join(root_dir, 'groundtruth')

_gather = scio.loadmat(split_path)    # dict_keys(['__header__', '__version__', '__globals__', 'testind', 'trainind'])
training = _gather['trainind'][0]-1  # 209 samples,
testing = _gather['testind'][0]-1    # 105 samples

name_list = os.listdir(img_dir)
name_list = np.array([i[:-4] for i in name_list])

train_name_list = name_list[training]
test_name_list = name_list[testing]

for name_ in test_name_list:
    _image_path = os.path.join(img_dir, name_ + '.jpg')
    _anno_path = os.path.join(anno_dir, name_ + '_labels.mat')

    _anno_data = scio.loadmat(_anno_path)   # dict_keys(['__header__', '__version__', '__globals__', 'fields', 'labels', 'gtPolyg'])
    _layout_seg = _anno_data['fields']  # (480, 640)
    _overall_seg = _anno_data['labels']  # (480, 640)
    _poly_vertex = _anno_data['gtPolyg']    # (1, k), k represents num_plane

    print(_poly_vertex)
    plt.imshow(_layout_seg)
    plt.show()





# root_dir1 = '/home/hui/database/dataset/zz_other_dataset/hedau/groundtruth/1BedRoom_lg_labels.mat'
# aaa = scio.loadmat(root_dir1)
# print(aaa.keys())

















