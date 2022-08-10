import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import skimage.io as io
import json
import scipy.io as scio
from search_polygon_predict import search_ploy, seg_color_trans

'''
代码作用：由Matterport3D-Layout数据集，生成’预测的’分割图、边缘对比图
不用单独生成graph.json, 只要提供数据集中对应标注数据的位置，一步到位，得到最后的结果
'''

## 1                                                seed(div)

# 2
# iname = '0aa01000f1934e73b35c348fa8d15040_i2_2'
# iname = '04cdd02138664b138f281bb5ad8b957f_i1_3'      #
# iname = '9b1fd662a14f4d07b1705ce2ed7adadb_i1_1'
# iname = '7ccfbff0b84147f48e041a1315c1a106_i2_2'
# iname = '9f69259571b54410997caf811a428107_i0_1'

## 3
# iname = '1cae8cb0d87e4c2694aabe3b9cecd0f9_i2_5'
# iname = '4e3ca18a1d6848529e1e97e841b3faa9_i1_0'
# iname = '6ab11272308c46068c9ae583da8ff311_i0_4'
# iname = '07e7a24c5ff449229aa0c72216829072_i1_2'
# iname = '0387739d97b34c12bb30593672a0b4ef_i1_2'
# iname = 'cd8460c3c9834ea0a1f61611f39d24a4_i0_4'
iname = 'c7af265804774a4cacbcb628542c348d_i1_5'        # 论文里的第二张
# iname = 'b2157583eedf426383de6e6e9753a4da_i2_4'

##  4
# iname = '2cd74c32749a45f5a81ebcd9d2f4c4c7_i0_1'
# iname = 'd6bccad7baed48caa7c559073ed8b3ad_i0_4'      # 论文里第三张
# iname = 'cd8460c3c9834ea0a1f61611f39d24a4_i1_2'
# iname = 'cd8460c3c9834ea0a1f61611f39d24a4_i0_2'
# iname = 'acc9d5c4eaa8426cb12d58d14e47c6f6_i1_3'        # 论文里第四章



# split = 'train'
# split = 'test'
split = 'validation'


shift_div = 100

seed = 128

# num_edge_list = [3, 4, 5]
num_edge_list = [3, 4, 5, 6]

poly_threshould = 0.95

line_width = 6

np.random.seed(seed)

# 源数据集
dataset_name = 'Matterport3D-Layout'
srce_dataset_dir = '/home/hui/database/dataset/matterport3d_layout'


split_path = os.path.join(srce_dataset_dir, split)

record_path = os.path.join(split_path, f'{split}.npz')
# ['image', 'layout_depth', 'layout_seg', 'depth', 'init_label', 'intrinsics_matrix',
# 'point', 'face', 'origin_params', 'params', 'nd_4', 'nd_5', 'group']
record_file = np.load(record_path, allow_pickle=True)
kp_list = record_file['point']
image_name_list = record_file['image']

# 从split.npz文件中找出对应图片的索引，进而取出对应的关键点坐标
for itr, i in enumerate(image_name_list):
    if i == (iname + '.jpg'):
        keypoint_3d = kp_list[itr]
        break

image_path = os.path.join(split_path, 'image', iname + '.jpg')
img_origin = io.imread(image_path)  # 原本大小
img = cv2.resize(img_origin, (512, 512))

seg_path = os.path.join(split_path, 'layout_seg', iname + '_seg.png')
layout_seg = io.imread(seg_path)
layout_seg_origin = layout_seg.copy()   # 原本大小
height, weight = layout_seg.shape

fx = 512/weight
fy = 512/height

# kp_path = os.path.join(split_path, 'layout_keypoint', iname + '.npy')
# keypoint_3d = np.load(kp_path)     # shape=(20, 3),包括了图片4个顶点,不够20个点则填充(0, 0, 0)到20个点
index_ = np.where((keypoint_3d != [0., 0., 0.]).any(1))[0]  #求出非填充点的索引（填充点3个坐标全是0）
corner = keypoint_3d[index_, 0:2]

corner_ = np.round(corner, 0)   #
non_border_index = []   # 取出corner的非图片角点的关键点，省的改代码
xxx_ = [0., weight-1., height-1.]
for itr, c_ in enumerate(corner_):
    if  not ((c_[0] in xxx_) and (c_[1] in xxx_)):
        non_border_index.append(itr)
corner = corner[non_border_index]   # 不包含图片角点的关键点

graph_anno_file = os.path.join(split_path, f'{split}.json')

border_point = np.array([[0, 0], [weight-1, 0], [0, height-1], [weight-1, height-1]]).astype(np.float64)
corner = np.concatenate([corner, border_point], axis=0)
corner_int = np.round(corner, 0)    # 未缩放的corner，转化为int
corner_scaled = corner.copy()

# corner_int_unscale = np.round(corner, 0)

# 对于落在图片边界上的点的分轴，不进行抖动，所以记录下索引.(但是LSUN中的交点标注又问的，有的交点=w，有的=w-1)
corner_1d = corner_int.reshape(-1) # 但是数据中的角点坐标保存的是小数，会比实际小一点（479保存为478.99987793），所以需要进行round
index_0 = np.where(corner_1d==0.)[0]
index_h = np.where(corner_1d==height-1.)[0]
index_h_ = np.where(corner_1d==height+0.)[0]
index_w = np.where(corner_1d==weight-1.)[0]
index_w_ = np.where(corner_1d==weight+0.)[0]
index_ = np.concatenate([index_0, index_h, index_w, index_h_, index_w_], axis=0)

num_junc = corner.shape[0]
# 偏移噪音
sign_matrix = np.ones((num_junc, 2))
p_or_n = np.random.normal(loc=0, scale=1, size=(num_junc, 2))   # 正太分布，决定噪音的正负，也既增大或减小
mask_n = p_or_n < 0
sign_matrix[mask_n] = -1.   # 决定噪音的正负，内部数值全是1或-1
shift_noise = 1 - (np.random.random((num_junc, 2))/shift_div) * sign_matrix     # 全体noise

# 给gt corner加上噪音，得到predcorner
moved_corner = corner * shift_noise

# 对落在图片边界上的点的分轴，保持不变
moved_corner_1d = moved_corner.reshape(-1)
moved_corner_1d[index_] = corner_1d[index_]
moved_corner = moved_corner_1d.reshape(-1, 2)

# 假设图片中的关键点为N，则corner4search.shape = (N,2),corner.shape = moved_corner.shape = (N+4, 2), 原本的h w，原本大小的layout_seg和img

trans_pred_seg, final_graph = search_ploy(corner, moved_corner, height, weight, layout_seg, img_origin,
                                          num_edge_list, poly_threshould, dataset_name)  # 未缩放至512的多边形
trans_pred_seg = cv2.resize(trans_pred_seg, (512, 512)) # (source_h, source_w)-->(512, 512)
trans_pred_seg = seg_color_trans(trans_pred_seg, dataset_name)

lines = final_graph
lines_scaled = lines.copy()

# lsun原本的标注中关键点坐标有问题，有些关键点的坐标等于了图片边长
lines_scaled[:, 0] = np.clip(lines[:, 0] * fx, 0, 511)
lines_scaled[:, 2] = np.clip(lines[:, 2] * fx, 0, 511)
lines_scaled[:, 1] = np.clip(lines[:, 1] * fy, 0, 511)
lines_scaled[:, 3] = np.clip(lines[:, 3] * fy, 0, 511)
gt_lines = lines_scaled.copy()
pred_lines = lines_scaled.copy()

# 缩放
corner_scaled[:, 0] = np.clip(corner[:, 0]*fx, 0, 511)
corner_scaled[:, 1] = np.clip(corner[:, 1]*fy, 0, 511)
moved_corner[:, 0] = np.clip(moved_corner[:, 0]*fx, 0, 511)
moved_corner[:, 1] = np.clip(moved_corner[:, 1]*fy, 0, 511)

corner_scaled_int = np.round(corner_scaled, 0)

pred_lines = pred_lines.reshape(-1, 2)
pred_lines_int = np.round(pred_lines, 0)


# 建立原始corner和lines拆分出来的lines_的联系
# 找出lines_中每一个点在corner中的索引，然后替换成moved_corner对应位置的点
for idx, point_ in enumerate(pred_lines_int):
    subindex = np.where((corner_scaled_int==point_).all(1))[0]
    pred_lines[idx] = moved_corner[subindex]

pred_lines = pred_lines.reshape(-1, 4)
pred_lines_plot = np.zeros_like(pred_lines)
pred_lines_plot[:, 0] = pred_lines[:, 0]
pred_lines_plot[:, 1] = pred_lines[:, 2]
pred_lines_plot[:, 2] = pred_lines[:, 1]
pred_lines_plot[:, 3] = pred_lines[:, 3]
pred_lines_plot = pred_lines_plot.reshape(-1, 2, 2)

gt_lines_plot = np.zeros_like(gt_lines)
gt_lines_plot[:, 0] = gt_lines[:, 0]
gt_lines_plot[:, 1] = gt_lines[:, 2]
gt_lines_plot[:, 2] = gt_lines[:, 1]
gt_lines_plot[:, 3] = gt_lines[:, 3]
gt_lines_plot = gt_lines_plot.reshape(-1, 2, 2)

plt.figure(figsize=(21, 7))
plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.imshow(trans_pred_seg.astype(np.int64))
plt.subplot(133)
plt.imshow(img)
for x, y in gt_lines_plot:
    plt.plot(x, y, color='r', linewidth=line_width)
for x, y in pred_lines_plot:
    plt.plot(x, y, color='g', linewidth=line_width)
plt.show()


# plt.figure(figsize=(5, 5))
# plt.imshow(img)
# for x, y in gt_lines_plot:
#     plt.plot(x, y, color='r', linewidth=line_width)
# for x, y in pred_lines_plot:
#     plt.plot(x, y, color='g', linewidth=line_width)
# plt.show()



save_dir = os.path.join('/home/hui/database/dataset/LCNN_Data/Experiment_LCNN',f'Experiment_on_{dataset_name}', 'lcnn', f'{iname}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_save_path = os.path.join(save_dir, f'image_{iname}.png')
seg_save_path = os.path.join(save_dir, f'seg_{iname}.png')
comparison_edge_save_path = os.path.join(save_dir, f'comparison_edge_{iname}.png')
gt_edge_save_path = os.path.join(save_dir, f'gt_edge_{iname}.png')
path_list = [img_save_path, seg_save_path, gt_edge_save_path, comparison_edge_save_path]

type_list = ['image', 'seg', 'gt_edge', 'comparison_edge']

for tp, path in zip(type_list, path_list):
    plt.figure(figsize=(5, 5))    # figsize中高宽的比例，要和待保存图片的高宽比例一致
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    if tp == 'image':
        plt.imshow(img)
        plt.savefig(img_save_path)
    if tp == 'seg':
        plt.imshow(trans_pred_seg.astype(np.int64))
        plt.savefig(seg_save_path)
    if tp== 'gt_edge':

        # 排除掉图片边界上的直线
        gt_lines_roomnet = []
        for l_ in gt_lines:
            if (l_[0] + l_[2] == 0 or l_[0] + l_[2] == 1022) or (l_[1] + l_[3] == 0 or l_[1] + l_[3] == 1022):
                continue
            gt_lines_roomnet.append(l_)

        gt_lines_roomnet = np.array(gt_lines_roomnet)
        num_line_valid, _ = gt_lines_roomnet.shape

        gt_lines_plot_ = np.zeros((num_line_valid, 4))
        gt_lines_plot_[:, 0] = gt_lines_roomnet[:, 0]
        gt_lines_plot_[:, 1] = gt_lines_roomnet[:, 2]
        gt_lines_plot_[:, 2] = gt_lines_roomnet[:, 1]
        gt_lines_plot_[:, 3] = gt_lines_roomnet[:, 3]
        gt_lines_plot_ = gt_lines_plot_.reshape(-1, 2, 2)

        plt.imshow(img)
        for x, y in gt_lines_plot_:
            plt.plot(x, y, color='r', linewidth=line_width)
        plt.savefig(gt_edge_save_path)

    if tp == 'comparison_edge':
        plt.imshow(img)
        for x, y in gt_lines_plot:
            plt.plot(x, y, color='r', linewidth=line_width)
        for x, y in pred_lines_plot:
            plt.plot(x, y, color='g', linewidth=line_width)
        plt.savefig(comparison_edge_save_path)
    # plt.show()









