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



root_path = '/home/hui/database/dataset/LSUN/'

split = 'training'
# split = 'validation'
# split = 'testing'

sample_range = [0, 4000]
# sample_range = [470, 471]
# sample_range = [60, 61]


threshould = 0.96

# 提取出来的信息的保存位置
output_save_path = os.path.join(root_path, split, split + f'_{sample_range[0]}_{sample_range[1]}_{threshould}' + '.json')

# 有问题图片的保存路径，需要单独处理
problem_save_path = os.path.join(root_path, split, split + '_problem' + f'_{sample_range[0]}_{sample_range[1]}_{threshould}' + '.txt')

anno_path = os.path.join(root_path, split, split + '.mat')     # mat文件保存了对应split的（图片名，房间类型，墙面数量，角落坐标）信息
anno_data = scio.loadmat(anno_path)

aggregate_data = []
problem_image = []


for sample_index in range(sample_range[0], sample_range[1]):
    print('Processimg image {a} ~'.format(a = sample_index))
    # 标注信息字典中的key是'validation'
    image_preffix = anno_data[split][0][sample_index][0][0]    # str
    room_type = anno_data[split][0][sample_index][1][0]        # str
    num_wall = anno_data[split][0][sample_index][2][0][0]      # int
    corners = anno_data[split][0][sample_index][3]             # 2D array, [[w, h]]

    h_w = anno_data[split][0][sample_index][4][0]              # [h, w]
    source_h = h_w[0]
    source_w = h_w[1]
    h = h_w[0]
    w = h_w[1]

    border_point = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]]).astype(np.float64)
    all_points_origin = np.concatenate([corners, border_point], axis=0)    # (u, v), 还是小数， 这个是不会变换的
    all_points_origin_int = np.round(all_points_origin, 0)    # 整数

    all_points = all_points_origin.copy()   # 小数， 待会可能会变换
    all_points_int = np.round(all_points, 0).astype(np.int64)  # 先四舍五入，再转化成整数

    seg_path = os.path.join(root_path, split, 'layout_seg_images', image_preffix + '.mat')
    layout_seg = scio.loadmat(seg_path)['layout']

    # 有的图片非常大，将图片先缩小再处理
    biggest_side = np.max([h, w])
    if biggest_side > 1000:
        # biggest_index = np.where((h_w == biggest_side))[0]
        scale_factor = 600 / biggest_side

        h_w = h_w * scale_factor
        h = round(h_w[0])
        w = round(h_w[1])
        all_points = all_points_origin * scale_factor
        all_points_int = np.round(all_points, 0).astype(np.int64)   #

        layout_seg = cv2.resize(layout_seg, (w, h))

    # print('image_name: ', image_preffix)
    # print('h_w of image:', h, w)
    # print('num_of_points:', len(all_points))
    # print('all_points:\n', all_points_int)
    # plt.imshow(layout_seg)
    # plt.show()

    '''
    sun_bhihhtiqlhpcpigw
    living_room
    5
    [[ 688.84883721  268.23837209]
     [   0.          496.79651163]
     [ 955.5           0.        ]
     [ 685.6744186  1039.62209302]
     [   0.         1001.52906977]
     [1023.         1187.23255814]]
    [1365 1023]
    '''


    polygon_candidate = PolygonCandidate(layout_seg, 'Lsun')

    # for plane_candi in permutations(all_points, 4):
    for NumVertexOfPolygon in [3, 4, 5, 6]:    # 一个多边形由几个顶点组成
    # for NumVertexOfPolygon in [4, 5]:    # 一个多边形由几个顶点组成
        for plane_candi in combinations(all_points_int, NumVertexOfPolygon):

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
    # 可能缩放int--未缩放int--未缩放float
    final_polygons_int, final_polygons_float = polygon_candidate.determine_final_polygon(all_points_origin, all_points_origin_int, all_points_int)

    if final_polygons_int == None:
        print(f'阈值设大了，改小点~~~{image_preffix}')
        # problem_image.append([split, sample_index, image_prefix + '.jpg', '阈值设大了，改小点'])

        with open(problem_save_path, 'a') as f:
            # f.truncate(0)
            f.write(str([split, sample_index, image_preffix + '.jpg', '阈值设大了，改小点']))
            f.write('\r\n')

    # polygon_candidate.poly_visualize()

    final_graph = polygon_candidate.generate_graph_CloseLoop()    # json不能保存ndarray


    if len(final_graph) == 0:
        print(f'阈值设小了，改大点~~~{image_preffix}')

        with open(problem_save_path, 'a') as f:
            # f.truncate(0)
            f.write(str([split, sample_index, image_preffix + '.jpg', '阈值设小了，改大点']))
            f.write('\r\n')
        continue

    final_graph = final_graph.tolist()

    final_dict = {'filename': image_preffix + '.jpg',   # 这是RGB图片的文件名，所以是.jpg(具体还是要看数据集)
                  'lines': final_graph,
                  'height': int(source_h),
                  'weight': int(source_w)
                  }

    aggregate_data.append(final_dict)



with open(output_save_path,'w') as file_obj:
    json.dump(aggregate_data, file_obj)

# with open(problem_save_path, 'w') as f:
#     f.truncate(0)
#     f.write(str(problem_image))




