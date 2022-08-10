import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.io as scio
import os
import skimage.io as io
from skimage.filters import  sobel
import copy

'''
输出大论文第一章的第一个图，说明布局有三种参数化方法。
'''


# iname = '4dcc93d2006646e19f9b2748dcb4f860_i1_3'
# iname = '0674ebc36a2048828d0ce0dc80751886_i1_3'
iname = '24e1d29d44014a1484f7b0e48bbaffe5_i1_3'

root_dir = '/home/hui/database/dataset/matterport3d_layout/validation'

image_path = os.path.join(root_dir, 'image', iname+'.jpg')
seg_path = os.path.join(root_dir, 'layout_seg', iname+'_seg.png')
anno_path = os.path.join('/home/hui/database/dataset/matterport3d_layout/record_file/validation.npz')

img = io.imread(image_path)
layout_seg = io.imread(seg_path)
h,w = layout_seg.shape  # 1024, 1280

data = np.load(anno_path, allow_pickle=True)

image_names = data['image']

for idx, i_ in enumerate(image_names):
    if i_ == (iname+'.jpg'):
        index_ = idx

keypoint = data['point'][index_]

layout_edge_ = sobel(layout_seg)
kernel = np.ones((3, 3), np.uint8)
layout_edge_ = cv2.dilate(layout_edge_, kernel, iterations=10)


color_dict = {'red': [255, 0, 0], 'green': [0, 255, 0], 'blue': [0, 0, 255], 'yellow': [255, 255, 0],
              'cyan': [0, 255, 255], 'purple': [138, 43, 226], 'gary': [192, 192, 192]}

layout_edge = np.zeros((h, w, 3))
edge_mask = layout_edge_>0.0001
un_edge_mask = layout_edge_ == 0

layout_edge[:, :, 0][edge_mask] = 255
layout_edge[:, :, 1][edge_mask] = 0
layout_edge[:, :, 2][edge_mask] = 0

layout_edge[:, :, 0][un_edge_mask] = 252
layout_edge[:, :, 1][un_edge_mask] = 230
layout_edge[:, :, 2][un_edge_mask] = 202


x_ = np.clip(keypoint[0], 0, w-1)
y_ = np.clip(keypoint[1], 0, h-1)

save_dir = os.path.join('/home/hui/database/dataset/dataset_show_others/three_representations', iname)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_save_path = os.path.join(save_dir, f'image_{iname}.png')
seg_save_path = os.path.join(save_dir, f'seg_{iname}.png')
edge_save_path = os.path.join(save_dir, f'edge_{iname}.png')
kp_save_path = os.path.join(save_dir, f'keypoint_{iname}.png')
path_list = [img_save_path, seg_save_path, edge_save_path, kp_save_path]


type_list = ['image', 'seg', 'edge', 'point']


for tp, path in zip(type_list, path_list):
    plt.figure(figsize=(5, 4))    # figsize中高宽的比例，要和待保存图片的高宽比例一致
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    if tp == 'image':
        plt.imshow(img)
        plt.savefig(img_save_path)
    if tp == 'seg':
        plt.imshow(layout_seg.astype(np.int64))
        plt.savefig(seg_save_path)
    if tp == 'edge':
        plt.imshow(layout_edge.astype(np.int64))
        plt.savefig(edge_save_path)
    else:
        pass
        plt.imshow(img)
        plt.scatter(x_, y_, color='r', s=500)
        plt.savefig(kp_save_path)
    # plt.show()
