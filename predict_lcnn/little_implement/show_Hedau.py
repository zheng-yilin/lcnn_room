import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.io as scio
import os
import skimage.io as io


# iname = '1BedRoom_lg'   # 3wall, cf
# iname = 'p1010067_c'   # 3wall, cf
# iname = '1_furniture'   #2 wall, f
# iname = '1456_3'   #2 wall, f
# iname = '0000000013'   #2 wall, f
iname = 'indoor_0375'   #2 wall, f


root_dir = '/home/hui/database/dataset/zz_other_dataset'

save_dir = os.path.join('/home/hui/database/dataset', 'dataset_show_others')

dataset_name = 'hedau'

image_path = os.path.join(root_dir, dataset_name, 'Images', iname+'.jpg')
anno_path = os.path.join(root_dir, dataset_name, 'groundtruth', iname+'_labels.mat')

img = io.imread(image_path)

anno_file = scio.loadmat(anno_path)
fields = anno_file['fields']
labels = anno_file['labels']
poly = anno_file['gtPolyg']

img = cv2.resize(img, (500, 500))
fields = cv2.resize(fields, (500, 500))
labels = cv2.resize(labels, (500, 500))

'''
keys(): ['__header__', '__version__', '__globals__', 'fields', 'labels', 'gtPolyg']
hedau的布局平面标签定义方式：完全情况（最复杂的情况）是3个墙面加天花板地面的盒型房间，2-前墙，4-左墙，3-右墙，1-地面，5-天花板。

fields为布局平面分割图，但是也有布局边缘，其标签为6。
labels:出现了室内空间物体，但是所有物体的标签都为6，布局边缘的标签变为7
gtPolyg:布局多边形的顶点信息。


'''


# print(fields.shape)
# print(fields)
# print(labels.shape)
# print(labels)
# print(poly.shape)
# print(poly)

# plt.subplot(131)
# plt.imshow(img)
# plt.subplot(132)
# plt.imshow(fields)
# plt.subplot(133)
# plt.imshow(labels)
#
# plt.show()

save_dir = os.path.join(save_dir, dataset_name, f'{iname}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_save_path = os.path.join(save_dir, f'image_{iname}.png')
fields_save_path = os.path.join(save_dir, f'fields_{iname}.png')
labels_save_path = os.path.join(save_dir, f'labels_{iname}.png')
path_list = [img_save_path, fields_save_path, labels_save_path]

data_list = [img, fields, labels]

for data, path in zip(data_list, path_list):
    plt.figure(figsize=(5, 5))  # figsize中高宽的比例，要和待保存图片的高宽比例一致
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.imshow(data)
    plt.savefig(path)
    # plt.show()



















