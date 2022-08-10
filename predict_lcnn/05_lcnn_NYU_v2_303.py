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
代码作用：由NYU v2 303数据集，生成’预测的’分割图、边缘对比图
不用单独生成graph.json, 只要提供数据集中对应标注数据的位置，一步到位，得到最后的结果

按IL标注方法标注出来的NYu v2 303数据集质量也较差，下面的代码失败
'''


## 1                         # seed
iname = '00464'      # 10027(120, 120)
# iname = '00549'        # 10023(130, 100)
# iname = '00588'       # 10023(110, 110)
# iname = '00868'       # 10023(70, 70)

## 2
# iname = '00354'       # 10023(210, 210)
# iname = '00415'       # 10023(180, 180)
# iname = '00832'         # 10023(200, 200)
# iname = '00976'           # 10027(120, 120) or (10035, 80, 80)



## 3
# iname = '00219'       # 10036(140, 140)
# iname = '00608'        # 10041(110, 110)
# iname = '00969'           # 10058(500, 500)   (10060, 800, 800)
# iname = '01150'          # 10073(210, 210)


##4
# iname = '01223'       # 10076(400, 400)


# split = 'train'
split = 'test'
# split = 'validation'


shift_div = 1000

seed = 118

poly_threshould = 0.96

line_width = 4

np.random.seed(seed)

# 源数据集
dataset_name = 'NYU_v2_303'
srce_dataset_dir = '/home/hui/database/dataset/nyu303_yao/nyu_303'

split_path = os.path.join(srce_dataset_dir, split)

image_path = os.path.join(split_path, 'image', iname + '.jpg')
img_origin = io.imread(image_path)  # 原本大小
img = cv2.resize(img_origin, (512, 512))

seg_path = os.path.join(split_path, 'layout_seg', iname + '.png')
layout_seg = io.imread(seg_path)
layout_seg_origin = layout_seg.copy()   # 原本大小
height, weight = layout_seg.shape

fx = 512/weight
fy = 512/height

kp_path = os.path.join(split_path, 'layout_keypoint', iname + '.npy')
keypoint_3d = np.load(kp_path)     # shape=(20, 3),包括了图片4个顶点,不够20个点则填充(0, 0, 0)到20个点
print(11111111111)
print(keypoint_3d)


index_ = np.where((keypoint_3d != [0., 0., 0.]).any(1))[0]  #求出非填充点的索引（填充点3个坐标全是0）
corner = keypoint_3d[index_, 0:2]
print(222)
print(corner)

corner_ = np.round(corner, 0)   #
non_border_index = []   # 取出corner的非图片角点的关键点，省的改代码
xxx_ = [0., weight-1., height-1.]
for itr, c_ in enumerate(corner_):
    if  not ((c_[0] in xxx_) and (c_[1] in xxx_)):
        non_border_index.append(itr)
corner = corner[non_border_index]   # 不包含图片角点的关键点

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

trans_pred_seg, final_graph = search_ploy(corner, moved_corner, height, weight, layout_seg, img_origin, poly_threshould, dataset_name)  # 未缩放至512的多边形
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

# print('*'*100)
# print(pred_lines_int)
# print('*'*100)
# print(corner_int)
# print('*'*100)
# print(moved_corner)


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

plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.imshow(trans_pred_seg.astype(np.int64))
plt.subplot(133)
plt.imshow(img)
for x, y in gt_lines_plot:
    plt.plot(x, y, color='r', linewidth='2')
for x, y in pred_lines_plot:
    plt.plot(x, y, color='g', linewidth='2')
plt.show()


# plt.figure(figsize=(5, 5))
# plt.imshow(img)
# for x, y in gt_lines_plot:
#     plt.plot(x, y, color='r', linewidth=line_width)
# for x, y in pred_lines_plot:
#     plt.plot(x, y, color='g', linewidth=line_width)
# plt.show()



save_dir = os.path.join('/home/hui/database/dataset/LCNN_Data/Experiment_LCNN',f'Experiment_on_{dataset_name}', f'{iname}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_save_path = os.path.join(save_dir, f'image_{iname}.png')
seg_save_path = os.path.join(save_dir, f'seg_{iname}.png')
comparison_edge_save_path = os.path.join(save_dir, f'comparison_edge_{iname}.png')
gt_edge_save_path = os.path.join(save_dir, f'gt_edge_{iname}.png')
path_list = [img_save_path, seg_save_path, gt_edge_save_path, comparison_edge_save_path]

type_list = ['image', 'seg', 'gt_edge', 'comparison_edge']
#
# for tp, path in zip(type_list, path_list):
#     plt.figure(figsize=(5, 5))    # figsize中高宽的比例，要和待保存图片的高宽比例一致
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#     plt.margins(0, 0)
#     if tp == 'image':
#         plt.imshow(img)
#         plt.savefig(img_save_path)
#     if tp == 'seg':
#         plt.imshow(trans_pred_seg.astype(np.int64))
#         plt.savefig(seg_save_path)
#     if tp== 'gt_edge':
#
#         # 排除掉图片边界上的直线
#         gt_lines_roomnet = []
#         for l_ in gt_lines:
#             if (l_[0] + l_[2] == 0 or l_[0] + l_[2] == 1022) or (l_[1] + l_[3] == 0 or l_[1] + l_[3] == 1022):
#                 continue
#             gt_lines_roomnet.append(l_)
#
#         gt_lines_roomnet = np.array(gt_lines_roomnet)
#         num_line_valid, _ = gt_lines_roomnet.shape
#
#         gt_lines_plot_ = np.zeros((num_line_valid, 4))
#         gt_lines_plot_[:, 0] = gt_lines_roomnet[:, 0]
#         gt_lines_plot_[:, 1] = gt_lines_roomnet[:, 2]
#         gt_lines_plot_[:, 2] = gt_lines_roomnet[:, 1]
#         gt_lines_plot_[:, 3] = gt_lines_roomnet[:, 3]
#         gt_lines_plot_ = gt_lines_plot_.reshape(-1, 2, 2)
#
#         plt.imshow(img)
#         for x, y in gt_lines_plot_:
#             plt.plot(x, y, color='r', linewidth=line_width)
#         plt.savefig(gt_edge_save_path)
#
#     if tp == 'comparison_edge':
#         plt.imshow(img)
#         for x, y in gt_lines_plot:
#             plt.plot(x, y, color='r', linewidth=line_width)
#         for x, y in pred_lines_plot:
#             plt.plot(x, y, color='g', linewidth=line_width)
#         plt.savefig(comparison_edge_save_path)
#     # plt.show()



