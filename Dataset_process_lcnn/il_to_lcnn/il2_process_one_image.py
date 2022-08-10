import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as scio
import json
import os
import skimage.io as io
from itertools import combinations
# from itertools import permutations
from collections import Counter
from pprint import pprint

from Dataset_process_lcnn.adjust_order_clockwise import adjust_pts_order
from Dataset_process_lcnn.search_poly_candidate import PolygonCandidate
from Dataset_process_lcnn.process_json_lcnn_one import process_to_graph_one

root_path = '/home/hui/database/dataset/interiornet_layout/'

# split = 'train'
# split = 'val'
split = 'test'

# sample_range = [1500, 2000]
# sample_range = [23, 24]   # 阈值设置太大了，识别不到多边形
# sample_range = [54, 55]   # 阈值设置小了，多边形重叠了
# sample_range = [57, 58]   # 图片分辨率非常高(2700, 3600)

sample_index = 128

threshould = 0.93

show = 0

# vertex_list = [4]
# vertex_list = [5]
# vertex_list = [3, 4]
vertex_list = [4, 5]

# vertex_list = [3, 4, 5]
# vertex_list = [4, 5, 6]
# vertex_list = [3, 4, 5, 6]

# 提取出来的信息的保存位置
output_save_path = os.path.join(root_path, split, split + f'_{sample_index}_{sample_index+1}_{threshould}' + '.json')
# 有问题图片的保存路径，需要单独处理
# problem_save_path = os.path.join(root_path, split, split + '_problem' + f'_{sample_index}_{sample_index+1}' + '.txt')

anno_path = os.path.join(root_path, split, f'interiornet_layout_{split}.npy')     # 存放的只有图片名
images_list = np.load(anno_path)
image_prefix = images_list[sample_index]

layout_seg_path = os.path.join(root_path, split, 'layout_seg', image_prefix + '.png')
corners_path = os.path.join(root_path, split, 'layout_keypoint', image_prefix + '.npy')
face_path = os.path.join(root_path, split, 'face', image_prefix + '.npy')

layout_seg = io.imread(layout_seg_path) # (20, 2), 没包括图片4个角点，关键点数量不够20个的补充到20个

corners_with_zero = np.load(corners_path)         # (9), 墙面数量不足9个的填充0
index_corner = np.where((corners_with_zero > np.array([0., 0., 0.])).any(1))
corners = corners_with_zero[index_corner, :2][0]    # (w, h)形式，只包括uv坐标，没有z值

layout_face = np.load(face_path)        # 图片中墙面的标签
index_face = layout_face != 0.
layout_face = layout_face[index_face]

h, w = layout_seg.shape

if show == 1:
    plt.imshow(layout_seg)
    plt.show()

aggregate_data = []
# problem_image = []

border_point = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]]).astype(np.float64)

all_points = np.concatenate([corners, border_point], axis=0)    # (u, v)
all_points_int = all_points.astype(np.int64)

all_points_origin = all_points.copy()
all_points_origin_int = all_points_origin.astype(np.int64)

# 有的图片非常大，要处理非常久，所以将图片先缩小再处理
# biggest_side = np.max([h, w])
# if biggest_side > 1000:
#     # biggest_index = np.where((h_w == biggest_side))[0]
#     scale_factor = 600 / biggest_side
#
#     h_w = h_w * scale_factor
#     h = int(h_w[0])
#     w = int(h_w[1])
#     all_points = all_points_origin * scale_factor
#     all_points_int = all_points.astype(np.int64)
#
#     layout_seg = cv2.resize(layout_seg, (w, h))

print('image_name: ', image_prefix)
# print('h_w of image:', h, w)
# print('num_of_points:', len(all_points))
# print('all_points:\n', all_points_int)
# plt.imshow(layout_seg)
# plt.show()

count = 0

polygon_candidate = PolygonCandidate(layout_seg, 'InteriorNet-Layout')

# for plane_candi in permutations(all_points, 4):
for NumVertexOfPolygon in vertex_list:    # 一个多边形由几个顶点组成
# for NumVertexOfPolygon in [4, 5]:    # 一个多边形由几个顶点组成
    for plane_candi in combinations(all_points_int, NumVertexOfPolygon):

        count += 1

        if count %2000 == 0:
            print(f'try~~~{count}~~~')

        plane_candi = adjust_pts_order(plane_candi)     # 将凸包的顶点顺序调整成顺时针方向，对凹多边形没用

        canvas = np.zeros((h, w), np.uint8)

        mask_ = cv2.fillConvexPoly(canvas, plane_candi, (101))
        mask = mask_ == 101
        CorrePartOfMask = layout_seg[mask]

        value_freq_dict = Counter(CorrePartOfMask)  # 取出的区域中出现的元素及其频次
        # print(value_time_dict)
        num_wall = len(value_freq_dict) # 出现的墙面个数

        # 如果取到同一条直线上的点，则舍此这次采样
        if num_wall == 0:
            continue

        key_wall = list(value_freq_dict.keys())     # 出现的墙面内标签
        value_wall = list(value_freq_dict.values()) # 每个墙面的像素数量

        NumPixelInArea = np.sum(value_wall)     # mask中所有像素数量
        value_of_max = np.max(value_wall)       # 出现的墙面中，面积最大墙面的像素数量
        index_of_max = value_wall.index(value_of_max)   # 最多像素数量的墙面在列表中的索引
        label_of_max = key_wall[index_of_max]   # 通过索引求出像素数量最多墙面的墙面标签

        max_ratio = value_of_max/NumPixelInArea # 最多像素的墙面

        if label_of_max in polygon_candidate.label_of_wall and (max_ratio >threshould):
            polygon_candidate.update(label_of_max, value_of_max, plane_candi)

final_polygons_int, final_polygons_float = polygon_candidate.determine_final_polygon(all_points_origin, all_points_origin_int, all_points_int)

if final_polygons_int == None:
    print('阈值设大了，改小点')
#     problem_image.append([split, sample_index, image_preffix + '.jpg', '阈值设大了，改小点'])


# print('~~~~~~~~~~~~~~~~~result~~~~~~~~~~~~~~~')
# pprint(final_polygons_int)
# pprint(final_polygons_float)

polygon_candidate.poly_visualize()

final_graph = polygon_candidate.generate_graph_CloseLoop()    # json不能保存ndarray
# print(111)
# print(final_graph)

if len(final_graph) == 0:
    print('阈值设小了，改大点')
    # problem_image.append([split, sample_index, image_preffix + '.jpg', '阈值设小了，改大点'])

final_graph = final_graph.tolist()
# print(final_graph)

final_dict = {'filename': image_prefix + '.jpg',   # 这是RGB图片的文件名，所以是.jpg(具体还是要看数据集)
              'lines': final_graph,
              'height': int(h),
              'weight': int(w)
              }
# print(final_dict)

aggregate_data.append(final_dict)

process_to_graph_one(root_path, split, aggregate_data)

# with open(output_save_path,'w') as file_obj:
#     json.dump(aggregate_data, file_obj)

# with open(problem_save_path, 'w') as f:
#     f.truncate(0)
#     f.write(str(problem_image))