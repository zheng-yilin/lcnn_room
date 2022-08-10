import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as scio
import json
import os
import skimage.io as io
from itertools import combinations
from itertools import permutations
from collections import Counter
from pprint import pprint

from Dataset_process_lcnn.adjust_order_clockwise import adjust_pts_order
from Dataset_process_lcnn.search_poly_candidate import PolygonCandidate



root_path = '/home/hui/database/dataset/interiornet_layout'

split = 'train'
# split = 'val'
# split = 'test'

sample_range = [2000, 4000]     # train
# sample_range = [0, 833]   # val
# sample_range = [0, 852]     # test

# sample_range = [0, 5]     #

threshould = 0.97

# 提取出来的信息的保存位置
output_save_path = os.path.join(root_path, split, split + f'_{sample_range[0]}_{sample_range[1]}_{threshould}' + '.json')


# 有问题图片的保存路径，需要单独处理
problem_save_path = os.path.join(root_path, split, split + '_problem' + f'_{sample_range[0]}_{sample_range[1]}_{threshould}' + '.txt')

if os.path.exists(problem_save_path):
    os.system('rm problem_save_path')

anno_path = os.path.join(root_path, split, f'interiornet_layout_{split}.npy')     # 存放的只有图片名
images_list = np.load(anno_path)

aggregate_data = []
problem_image = []

for sample_index in range(sample_range[0], sample_range[1]):
    print('Processimg image {a} ~'.format(a = sample_index))
    # 标注信息字典中的key是'validation'
    image_prefix = images_list[sample_index]

    layout_seg_path = os.path.join(root_path, split, 'layout_seg', image_prefix + '.png')
    layout_seg = io.imread(layout_seg_path)
    h, w = layout_seg.shape

    corners_path = os.path.join(root_path, split, 'layout_keypoint', image_prefix + '.npy')

    corners_with_zero = np.load(corners_path)  # (20, 2), 包括了图片4个角点，关键点数量不够20个的补充到20个
    index_corner = np.where((corners_with_zero > np.array([0., 0., 0.])).any(1))
    all_points_origin = corners_with_zero[index_corner, :2][0]  # (w, h)形式，只包括uv坐标，没有z值;读取出来的corner包括4个角点
    all_points_origin_int = np.round(all_points_origin, 0).astype(np.int64)

    all_points = all_points_origin.copy()
    all_points_int = np.round(all_points, 0).astype(np.int64)

    polygon_candidate = PolygonCandidate(layout_seg, 'InteriorNet-Layout')

    # for plane_candi in permutations(all_points, 4):
    # for NumVertexOfPolygon in [3, 4, 5]:    # 一个多边形由几个顶点组成
    for NumVertexOfPolygon in [3, 4, 5, 6]:    # 一个多边形由几个顶点组成
        for plane_candi in combinations(all_points_int, NumVertexOfPolygon):
            # count += 1
            # print(count)

            plane_candi = adjust_pts_order(plane_candi)     # 将凸包的顶点顺序调整成顺时针方向，对凹多边形没用

            canvas = np.zeros((h, w), np.uint8)

            mask_ = cv2.fillConvexPoly(canvas, plane_candi, (1))
            mask = mask_ == 1
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
        print(f'阈值设大了，改小点~~~{image_prefix}')
        # problem_image.append([split, sample_index, image_prefix + '.jpg', '阈值设大了，改小点'])

        with open(problem_save_path, 'a') as f:
            # f.truncate(0)
            f.write(str([split, sample_index, image_prefix + '.jpg', '阈值设大了，改小点']))
            f.write('\r\n')

        continue

    # print('~~~~~~~~~~~~~~~~~result~~~~~~~~~~~~~~~')
    # pprint(final_polygons_int)
    # pprint(final_polygons_float)

    # polygon_candidate.poly_visualize()

    final_graph = polygon_candidate.generate_graph_CloseLoop()    # json不能保存ndarray
    # print(111)
    # print(final_graph)

    if len(final_graph) == 0:
        print(f'阈值设小了，改大点~~~{image_prefix}')
        # problem_image.append([split, sample_index, image_prefix + '.jpg', '阈值设小了，改大点'])

        with open(problem_save_path, 'a') as f:
            # f.truncate(0)
            f.write(str([split, sample_index, image_prefix + '.jpg', '阈值设小了，改大点']))
            f.write('\r\n')
        continue

    final_graph = final_graph.tolist()
    # print(final_graph)

    final_dict = {'filename': image_prefix + '.jpg',   # 这是RGB图片的文件名，所以是.jpg(具体还是要看数据集)
                  'lines': final_graph,
                  'height': int(h),
                  'weight': int(w)
                  }
    # print(final_dict)
    aggregate_data.append(final_dict)

with open(output_save_path,'w') as file_obj:
    json.dump(aggregate_data, file_obj)





