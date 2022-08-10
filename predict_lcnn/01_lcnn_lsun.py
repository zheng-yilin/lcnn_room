import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import skimage.io as io
import json
import scipy.io as scio
from search_polygon_predict import search_ploy, seg_color_trans
from skimage.filters import sobel


'''
代码作用：
（1）出lcnn-room方法和RommNet方法在LSUN数据集上的效果图
注:当method='roomnet'时，画出的edge_embedding图（pred edge与gt edge比较）不包含落在图片边界上的图
   当method='lcnn'时，画出的edge_embedding图包含落在图片边界上的图
'''


# 1 wall                                              # seed   lcnn       roomnet
# iname = 'sun_bbrgcasvzdvryynk'                               #116(27)   115(15)
# iname = '1d55e3914a87c8c4f5d911c70ac82e31474f1d7f'           #117(12)       121(18)

# 2wall
# iname = '01fe1b8ceed678b27fd45aca2bde6b8c7f0cad40'             #124(35)     126(27)
# iname = '82977512d3b886eadf97fc3325ce2ed830c05e1b'             #140(27)     139(22)
# iname = 'sun_aawlhrzlsdywldmq'                                 #143(15)     146(22)
# iname = 'sun_acddebisonjmjrum'
# iname = 'sun_afywcuzscjctbdcu'
# iname = '0aa0b07b9fc1d7f59c55b585be3ee3a3e77f31fc'

# 3wall
# iname = '2c4a7c9e03e1d61ab2975a6df13b39b6ff6f4a6e'              #154(25)    163(25)
# iname = '7e072d7ebc2cb048db52e25ea20322800ff8cf93'                # 1003(23)    1011(20)
# iname = 'sun_aclmoxwmazctsgls'                                    # 1010(28)  1011(20)
iname = 'sun_aiqzgxphqrsranga'                                      # 1012(18)  1015(25)

# iname = 'sun_ascscizerrcsswcd'                                      #
# iname = 'sun_ausqyahjtatjoasv'                                      #
# iname = 'sun_awzzrvilsjtdijwq'                                      #(1015, 25)



# method = 'lcnn'
method = 'roomnet'

seed = 1015

shift_div = 25

num_edge_list = [3, 4, 5]
# num_edge_list = [3, 4, 5, 6]

poly_threshould = 0.95

line_width = 6

np.random.seed(seed)

# 源数据集
dataset_name = 'Lsun'
srce_dataset_dir = '/home/hui/database/dataset/LSUN'

split = 'validation'

split_path = os.path.join(srce_dataset_dir, split)

lsun_anno_file = os.path.join(split_path, f'{split}.mat')
lsun_anno = scio.loadmat(lsun_anno_file)

graph_anno_file = os.path.join(split_path, f'{split}.json')
with open(graph_anno_file, "r") as f:
    # 对于每张图片，包括图片名、直线交点uv坐标（w,h,w,h）、图片高、图片宽,keys: filename(不包含_0), lines, height, weight
    graph_anno = json.load(f)

image_path = os.path.join(split_path, 'images', iname + '.jpg')
img_origin = io.imread(image_path)
img = cv2.resize(img_origin, (512, 512))
# plt.imshow(img)
# plt.show()
seg_path = os.path.join(split_path, 'layout_seg_images', iname + '.mat')
layout_seg = scio.loadmat(seg_path)['layout']
# plt.imshow(layout_seg)
# plt.show()

for idx in range(394):
    if lsun_anno[split][0][idx][0][0] == iname:
        # idx_lsun = idx
        data_lsun = lsun_anno[split][0][idx]

for idx, s_sample in enumerate(graph_anno):
    # print(s_sample['filename'])
    if s_sample['filename'][:-4] == iname:
        # idx_graph = idx
        data_graph = graph_anno[idx]

height = data_graph['height']
weight = data_graph['weight']
fx = 512/weight
fy = 512/height
lines = np.array(data_graph['lines'])

corner = data_lsun[3]   # (w, h)

border_point = np.array([[0, 0], [weight-1, 0], [0, height-1], [weight-1, height-1]]).astype(np.float64)
corner = np.concatenate([corner, border_point], axis=0)
# 对于落在图片边界上的点的分轴，不进行抖动，所以记录下索引.(但是LSUN中的交点标注又问的，有的交点=w，有的=w-1)
corner_1d = corner.reshape(-1)
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
# print('shift noise')
# print(shift_noise)
# 给gtcorner加上噪音，得到predcorner
moved_corner = corner * shift_noise

# 对落在图片边界上的点的分轴，保持不变
moved_corner_1d = moved_corner.reshape(-1)
moved_corner_1d[index_] = corner_1d[index_]
moved_corner = moved_corner_1d.reshape(-1, 2)

# 假设图片中的关键点为N，则corner4search.shape = (N,2),corner.shape = moved_corner.shape = (N+4, 2), 原本的h w，原本大小的layout_seg和img
trans_pred_seg, final_graph = search_ploy(corner, moved_corner, height, weight, layout_seg, img_origin,
                                          num_edge_list, poly_threshould, dataset_name)
trans_pred_seg = cv2.resize(trans_pred_seg, (512, 512)) # (source_h, source_w)-->(512, 512)
trans_pred_seg = seg_color_trans(trans_pred_seg, dataset_name)


# lsun原本的标注中关键点坐标有问题，有些关键点的坐标等于了图片边长
lines[:, 0] = np.clip(lines[:, 0] * fx, 0, 511)
lines[:, 2] = np.clip(lines[:, 2] * fx, 0, 511)
lines[:, 1] = np.clip(lines[:, 1] * fy, 0, 511)
lines[:, 3] = np.clip(lines[:, 3] * fy, 0, 511)
# print(lines)
# graph2seg(lines)

# 用来画 GT
gt_lines = lines.copy()

# 缩放
corner[:, 0] = np.clip(corner[:, 0]*fx, 0, 511)
corner[:, 1] = np.clip(corner[:, 1]*fy, 0, 511)
moved_corner[:, 0] = np.clip(moved_corner[:, 0]*fx, 0, 511)
moved_corner[:, 1] = np.clip(moved_corner[:, 1]*fy, 0, 511)

pred_lines = lines.reshape(-1, 2)

# 建立原始corner和lines拆分出来的lines_的联系
# 找出lines_中每一个点在corner中的索引，然后替换成moved_corner对应位置的点
for idx, point_ in enumerate(pred_lines):
    subindex = np.where((corner==point_).all(1))[0]
    pred_lines[idx] = moved_corner[subindex]

pred_lines = pred_lines.reshape(-1, 4)


if method == 'roomnet':
    # 如果是roomnet的图，就把图像边界上的线删除掉
    gt_lines_roomnet = []
    for l_ in gt_lines:
        if (l_[0]+l_[2] == 0 or l_[0]+l_[2] == 1022) or (l_[1]+l_[3] == 0 or l_[1]+l_[3] == 1022):
            continue
        gt_lines_roomnet.append(l_)

    pred_lines_roomnet = []
    for l_ in pred_lines:
        if (l_[0]+l_[2] == 0 or l_[0]+l_[2] == 1022) or (l_[1]+l_[3] == 0 or l_[1]+l_[3] == 1022):
            continue
        pred_lines_roomnet.append(l_)
    gt_lines_roomnet = np.array(gt_lines_roomnet)
    pred_lines_roomnet = np.array(pred_lines_roomnet)


    gt_lines = gt_lines_roomnet
    pred_lines = pred_lines_roomnet

gt_lines_plot = np.zeros_like(gt_lines)
gt_lines_plot[:, 0] = gt_lines[:, 0]
gt_lines_plot[:, 1] = gt_lines[:, 2]
gt_lines_plot[:, 2] = gt_lines[:, 1]
gt_lines_plot[:, 3] = gt_lines[:, 3]
gt_lines_plot = gt_lines_plot.reshape(-1, 2, 2)

pred_lines_plot = np.zeros_like(pred_lines)
pred_lines_plot[:, 0] = pred_lines[:, 0]
pred_lines_plot[:, 1] = pred_lines[:, 2]
pred_lines_plot[:, 2] = pred_lines[:, 1]
pred_lines_plot[:, 3] = pred_lines[:, 3]
pred_lines_plot = pred_lines_plot.reshape(-1, 2, 2)


# plt.figure(figsize=(5, 5))
# plt.imshow(img)
# for x, y in gt_lines_plot:
#     plt.plot(x, y, color='r', linewidth=line_width)
# for x, y in pred_lines_plot:
#     plt.plot(x, y, color='g', linewidth=line_width)
# plt.show()

plt.figure(figsize=(24, 8))
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


save_dir = os.path.join('/home/hui/database/dataset/LCNN_Data/Experiment_LCNN/Experiment_on_LSUN', method, f'{iname}')
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
    else:
        # cv2.imwrite(filename, )
        plt.imshow(img)
        for x, y in gt_lines_plot:
            plt.plot(x, y, color='r', linewidth=line_width)
        for x, y in pred_lines_plot:
            plt.plot(x, y, color='g', linewidth=line_width)
        plt.savefig(edge_save_path)
    # plt.show()







