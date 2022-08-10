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
代码作用：
（1）出lcnn-room方法在InteriorNet-Layout数据集上的效果图
（2）输出数据集展示中graph图
'''

# dataset show images
# 0：进行预测，1：画graph
# plot_graph = 1  # 是否出数据集展示里graph的图
# 3 wall
# iname = '1201_1301_3FO4IGAM4EVB_Kitchen_8'  # train(0, 2000)
# iname = '1509_1602_3FO4II469FHL_Dining_room_6'  # train(0, 2000)
iname = '1805_1901_3FO4IIX4QBH9_Dining_room_9'  # 最小环检索示意图
# 4 wall
# iname = '1012_1109_3FO4IFC5UYLB_Bedroom_9'  # train(4000, 6534)
# iname = '1201_1301_3FO4IGAM4EVB_Kitchen_2'  # train(0, 2000)
# iname = '1302_1403_3FO4IHKEUIVQ_Bedroom_14'  # train(0, 2000)
# 5 wall
# iname = '1201_1301_3FO4IGCSFPO8_Dining_room_19'  # train(4000, 6534)
# iname = '2105_2203_3FO4IJQ4QHBH_Dining_room_16'  # train(0, 2000)   # 我的算法(layout+corner-->多边形)失败案例，
# iname = '3005_3100_3FO4IMWOXYKF_Dining_room_13'  # train(0, 2000)
# iname = '3702_3804_3FO4INN6W55W_Dining_room_0'  # train(2000, 4000)



plot_graph = 0

## 1
# iname = '702_806_3FO4IDK3VBJX_Bedroom_10'

## 2
# iname = '1805_1901_3FO4IIX4QBH9_Bedroom_0'          # 136, 15
# iname = '1902_2005_3FO4IJ3DIL0Y_Dining_room_8'      # 137, 9
# iname = '3702_3804_3FO4INRSTK8V_Dining_room_10'       # 142, 25
# iname = '1708_1804_3FO4IIUUU4AG_Dining_room_0'          # 142, 20

## 3
# iname = '910_1011_3FO4IEOU1EVS_Kitchen_2'          # 失败案例
iname = '1509_1602_3FO4II8OLSP6_Dining_room_6'          # 两篇论文的流程框架图

# iname = '807_909_3FO4IDX9JIBX_Bathroom_2'
# iname = '1902_2005_3FO4IJ4H6J4L_Living_room_10'   # 105, 25
# iname = '2607_2702_3FO4ILQKSCNY_Dining_room_6'      # 134, 25


##4
# test
# iname = '1110_1200_3FO4IG9MC85A_Bathroom_0'   # 111, 23
# iname = '702_806_3FO4IDK3VBJX_Bedroom_6'        # 113, 17


# train
# iname = '807_909_3FO4IE5YY12I_Dining_room_12'       # 4000-6534, 131, 35
# iname = '807_909_3FO4IE2B8QR4_Dining_room_15'         # 2000-4000,  115, 23


## 5
# iname = '1404_1508_3FO4II5NRRSE_Bathroom_0'
# iname = '1404_1508_3FO4II5NRRSE_Bathroom_18'
# iname = '1509_1602_3FO4II8OLSP6_Dining_room_0'
# iname = '1805_1901_3FO4IIX4QBH9_Dining_room_18'   # 118, 20
# iname = '2006_2104_3FO4IJ5M525U_Living_room_15'
# iname = '2607_2702_3FO4ILQPUNJ0_Dining_room_18'
# iname = '2703_2802_3FO4IM3LQFN8_Dining_room_0'    # 118, 20
# iname = '2703_2802_3FO4IM60TMJG_Living_room_8'
# iname = '4506_4601_3FO4IOESYMQE_Balcony_15'
# iname = '5106_5205_3FO4IPE87S0Q_Living_room_6'

# split = 'train'
split = 'test'
# split = 'validation'


shift_div = 20

seed = 118

poly_threshould = 0.96

line_width = 4

np.random.seed(seed)

# 源数据集
dataset_name = 'InteriorNet-Layout'
srce_dataset_dir = '/home/hui/database/dataset/interiornet_layout'


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
# 因为trian中图片很多，所以分成三次从原本的标注信息转化为graph标注信息
# graph_anno_file = os.path.join(split_path, f'{split}_1.json') # 0-2000
# graph_anno_file = os.path.join(split_path, f'{split}_2.json')   # 2000-4000
# graph_anno_file = os.path.join(split_path, f'{split}_3.json') # 4000-6534

# with open(graph_anno_file, "r") as f:
#     # 对于每张图片，包括图片名、直线交点uv坐标（w,h,w,h）、图片高、图片宽,keys: filename(不包含_0), lines, height, weight
#     graph_anno = json.load(f)
#
# for idx, s_sample in enumerate(graph_anno):
#     if s_sample['filename'][:-4] == iname:
#         data_graph = graph_anno[idx]
#         # print(data_graph)
#
# lines = np.array(data_graph['lines'])
# lines_scaled = lines.copy()

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

if plot_graph == 1:
    trans_pred_seg = np.zeros((512, 512))  # 出graph图的时候用
else:
    trans_pred_seg = search_ploy(corner, moved_corner, height, weight, layout_seg, img_origin, poly_threshould, dataset_name)  # 未缩放至512的多边形
    trans_pred_seg = cv2.resize(trans_pred_seg, (512, 512)) # (source_h, source_w)-->(512, 512)
    trans_pred_seg = seg_color_trans(trans_pred_seg)



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

# plt.subplot(131)
# plt.imshow(img)
# plt.subplot(132)
# plt.imshow(trans_pred_seg.astype(np.int64))
# plt.subplot(133)
# plt.imshow(img)
# for x, y in gt_lines_plot:
#     plt.plot(x, y, color='r', linewidth='2')
# for x, y in pred_lines_plot:
#     plt.plot(x, y, color='g', linewidth='2')
# plt.show()


# plt.figure(figsize=(5, 5))
# plt.imshow(img)
# for x, y in gt_lines_plot:
#     plt.plot(x, y, color='r', linewidth=line_width)
# for x, y in pred_lines_plot:
#     plt.plot(x, y, color='g', linewidth=line_width)
# plt.show()



if plot_graph == 1:
    corner_int_graph = []  # 保存graph中的顶点（有些图片顶点可能不在graph中）
    gt_lines_int = np.round(lines).reshape(-1, 2)  # 将为进行缩放的lines转化为int

    for c_ in corner_int:
        sub_index = np.where((gt_lines_int == c_).all(1))[0]
        if len(sub_index) != 0:
            corner_int_graph.append(c_)
    corner_int_graph = np.array(corner_int_graph)

    gt_lines_us_plot = np.zeros_like(lines)
    gt_lines_us_plot[:, 0] = lines[:, 0]
    gt_lines_us_plot[:, 1] = lines[:, 2]
    gt_lines_us_plot[:, 2] = lines[:, 1]
    gt_lines_us_plot[:, 3] = lines[:, 3]
    gt_lines_us_plot = gt_lines_us_plot.reshape(-1, 2, 2)

    x_corner = corner_int_graph[:, 0]
    y_corner = corner_int_graph[:, 1]

    graph_save_dir = os.path.join(os.path.join(srce_dataset_dir, 'graph_paper', 'dataset_show_graph', iname))
    if not os.path.exists(graph_save_dir):
        os.makedirs(graph_save_dir)
    graph_save_path = os.path.join(graph_save_dir, f'graph_{iname}.png')


    plt.figure(figsize=(6.4, 4.8))
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(img_origin)
    for x, y in gt_lines_us_plot:
        plt.plot(x, y, color='g', linewidth='5')
    plt.scatter(x_corner, y_corner, color='r', s=300)
    # plt.savefig(graph_save_path)
    plt.show()
else:
    save_dir = os.path.join(srce_dataset_dir, 'graph_paper','il_lcnn', f'{iname}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_save_path = os.path.join(save_dir, f'image_{iname}.png')
    seg_save_path = os.path.join(save_dir, f'seg_{iname}.png')
    edge_save_path = os.path.join(save_dir, f'edge_{iname}.png')
    path_list = [img_save_path, seg_save_path, edge_save_path]

    type_list = ['image', 'seg', 'edge']

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
        if tp == 'edge':
            # cv2.imwrite(filename, )
            plt.imshow(img)
            for x, y in gt_lines_plot:
                plt.plot(x, y, color='r', linewidth=line_width)
            for x, y in pred_lines_plot:
                plt.plot(x, y, color='g', linewidth=line_width)
            plt.savefig(edge_save_path)
        # plt.show()









