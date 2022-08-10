import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import cv2
from itertools import combinations
import copy
from skimage.filters import sobel

from search_poly_candidate import PolygonCandidate
from adjust_order_clockwise import adjust_pts_order
from Hungarian_algorithm import match_trans_by_Hungarian


def edge_embedding(image_, layout_seg, color):
    # detect the layout edge, and plot replace the corrsponding pixel in the image
    image = copy.copy(image_)
    color_dict = {'red': [255, 0, 0], 'green': [0, 255, 0], 'blue': [0, 0, 255], 'yellow': [255, 255, 0],
                  'cyan':[0, 255, 255], 'purple': [138, 43, 226], 'gary': [192, 192, 192]}
    color_value = color_dict[color]

    layout_edge = sobel(layout_seg)
    gt_edge_mask = layout_edge>0.0001

    image[:, :, 0][gt_edge_mask] = color_value[0]
    image[:, :, 1][gt_edge_mask] = color_value[1]
    image[:, :, 2][gt_edge_mask] = color_value[2]

    return image

def seg_color_trans(layout_seg, dataset_name):
    # color_dict = {0: [0, 0, 255], 1: [127, 255, 0], 2: [0, 255, 255], 3:[255, 255, 0],
    #               4: [227, 23 ,13], 5: [218, 112, 214], 6: [255, 153, 87]}
    # il-lcnn是通过匈牙利算法和最初的语义标签（1，3，4，5，6，7，8）进行匹配，所以要对key进行一下替换

    if dataset_name == 'Lsun':
        key_list = [1, 2, 3, 4, 5]
        color_dict = {1:[0, 255, 255], 2:[255, 255, 0], 3:[227, 23, 13], 4:[127, 255, 0], 5:[0, 0, 255] }
    else:
        key_list = [1, 3, 4, 5, 6, 7, 8]
        color_dict = {1: [0, 0, 255], 3: [127, 255, 0], 4: [0, 255, 255], 5: [255, 255, 0],
                      6: [227, 23, 13], 7: [218, 112, 214], 8: [255, 153, 87]}

    # color_dict = {0: [0, 0, 255], 1: [127, 255, 0], 2: [0, 255, 255], 3: [255, 255, 0],
    #               4: [227, 23, 13], 5: [218, 112, 214], 6: [218, 112, 214]}


    h, w = layout_seg.shape
    trans_seg = np.zeros((h, w, 3))

    for i in key_list:
        if i not in layout_seg:
            continue
        color_value = color_dict[i]
        mask = layout_seg == i

        trans_seg[:, :, 0][mask] = color_value[0]
        trans_seg[:, :, 1][mask] = color_value[1]
        trans_seg[:, :, 2][mask] = color_value[2]

    return trans_seg.astype(np.int64)

def search_ploy(all_points_origin, moved_corner, h, w, layout_seg, img, threshould, dataset_name):
    '''
    # 结合layout_seg和gt corner，检测表示墙面的环。然后将每个环中的gt corner替换为pred corner。
    依次画出每个墙面的范围，并求出天花板与地面的范围。利用匈牙利算法进行语义转换
    '''
    layout_seg_origin = layout_seg.copy()   # 如果图片太大的话，会对layout_seg进行缩放，所以这里先保存一份原大小的

    source_h = h
    source_w = w

    h_w = np.array([h, w])

    # all_points_origin = corner  # 包含关键点+图片角点，小数
    all_points_origin_int = np.round(all_points_origin, 0)    # 整数
    all_points = all_points_origin.copy()   # 小数， 待会可能会变换
    all_points_int = np.round(all_points, 0).astype(np.int64)  # 先四舍五入，再转化成整数
    # seg_path = os.path.join(root_path, split, 'layout_seg_images', image_preffix + '.mat')
    # layout_seg = scio.loadmat(seg_path)['layout']

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
    polygon_candidate = PolygonCandidate(layout_seg, dataset_name)
    count = 0

    # for plane_candi in permutations(all_points, 4):
    for NumVertexOfPolygon in [3, 4, 5]:    # 一个多边形由几个顶点组成
    # for NumVertexOfPolygon in [4, 5]:    # 一个多边形由几个顶点组成
        for plane_candi in combinations(all_points_int, NumVertexOfPolygon):
            count += 1
            if count % 2000 == 0:
                print(count)

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
    # print(final_polygons_int)
    # print(final_polygons_float)
    '''
    {2: array([[785.4763491 , 500.99933378],
           [282.47834777, 750.        ],
           [  0.        , 749.        ],
           [  0.        ,   0.        ],
           [773.48434377, 151.89873418]]), 
    3: array([[1000.        ,  530.9793471 ],
           [ 785.4763491 ,  500.99933378],
           [ 773.48434377,  151.89873418],
           [1000.        ,  116.58894071]])}
    '''

    # 将多边形中的gt顶点替换为moved_corner
    final_polygons_moved = []
    num_wall = len(final_polygons_float.values())
    for i in range(num_wall):
        final_polygons_moved.append([])

    for iter_, poly_ in enumerate(final_polygons_int.values()):
        for vertex_ in poly_:
            idx_ = np.where((all_points_origin_int==vertex_ ).all(1))[0]
            if len(idx_) != 0:
                final_polygons_moved[iter_].append(moved_corner[idx_][0])   # moved_corner[idx_] = [[248.14824027 381.1973821 ]]
    # final_ploygons_moved = np.array(final_ploygons_moved)

    # 求出顶点漂移后的，每个墙面的mask以及所有墙面mask的并集
    wall_mask_integrete = np.zeros((source_h, source_w))
    wall_mask_seperate = []
    # for iter_, poly_ in enumerate(final_polygons_int):
    for iter_, poly_ in enumerate(final_polygons_moved):
        poly_ = np.array(poly_, dtype=np.int64)
        canvas = np.zeros((source_h, source_w), np.uint8)
        _mask = cv2.fillConvexPoly(canvas, poly_, 1)
        wall_mask_seperate.append(_mask)
        wall_mask_integrete += _mask


    # 墙面的mask之间会有交叠，所以求出交叠区域
    overlap_mask = np.zeros((source_h, source_w))
    for i in range(num_wall-1):
        overlap_mask += (wall_mask_seperate[i]+wall_mask_seperate[i+1])==2

    wall_centers = []
    for poly_ in final_polygons_moved:
        poly_ = np.array(poly_)
        center_x = np.mean(poly_[:, 0])
        center_y = np.mean(poly_[:, 1])
        wall_centers.append([center_x, center_y])
    wall_centers = np.array(wall_centers)

    # 将所有墙面中心按x坐标由小到达排序
    sort_index = np.argsort(wall_centers[:, 0], axis=0)
    wall_centers = wall_centers[sort_index]
    # print(wall_centers)

    left_point = np.array([[0, wall_centers[0][1]]])
    right_point = np.array([[source_w-1, wall_centers[-1][1]]])
    # 墙面中心连线。确保能将天花板与地面分隔开
    split_line = np.concatenate([left_point, wall_centers, right_point], axis=0)
    # print(left_point, right_point)
    # print(split_line)

    # plt.subplot(121)
    # plt.imshow(img)
    # plt.plot(split_line[:, 0],split_line[:, 1], color='r')
    # plt.subplot(122)
    # plt.imshow(layout_seg_origin)
    # plt.plot(split_line[:, 0],split_line[:, 1], color='r')
    # plt.show()

    left_up_point = np.array([[0., 0.]])
    right_up_point = np.array([[source_w-1., 0.]])
    # left_down_point = np.array([[0., source_h-1.]])
    # right_down_point = np.array([[source_w-1., source_h-1.]])

    # 包含天花板的多边形的顶点坐标
    up_poly = np.concatenate([split_line, right_up_point, left_up_point], axis=0).astype(np.int64)
    # down_poly = np.concatenate([split_line, right_down_point, left_down_point], axis=0).astype(np.int64)

    canvas = np.zeros((source_h, source_w), np.uint8)
    mask_ceil = cv2.fillPoly(canvas, [up_poly], 1)

    mask_floor = 1 - mask_ceil

    overlap_mask_ = overlap_mask>0.001
    # 求出地面的mask
    intersc_floor = (wall_mask_integrete + mask_floor)>=2
    floor_ = mask_floor - intersc_floor
    floor_[floor_ != 0]=1

    # 求出天花板的mask
    intersc_ceil = (wall_mask_integrete + mask_ceil)>=2
    ceil_ = mask_ceil - intersc_ceil
    ceil_[ceil_ != 0]=1

    layout_plane_masks = wall_mask_seperate.copy()  # 包括地面与天花板的分割掩模
    # 检测是否有天花板或者地面存在
    for m_ in [floor_, ceil_]:
        if np.sum(m_)/((source_h-1)*(source_w-1))>0.05:
            layout_plane_masks.append(m_)

    pred_seg = np.zeros((source_h, source_w))
    color = 0
    for m_ in layout_plane_masks:
        color+=1
        pred_seg += m_*color
    # plt.imshow(pred_seg)
    # plt.show()

    # 匈牙利算法进行语义转换
    PA, trans_pred_seg = match_trans_by_Hungarian(layout_seg_origin, pred_seg.astype(np.int))

    # 对于墙面交叠区域，用该像素位置的右边两个像素的值来取代
    nonzero_h, nonzero_w = np.nonzero(overlap_mask)
    trans_pred_seg[nonzero_h, nonzero_w] = trans_pred_seg[nonzero_h, nonzero_w+2]
    # plt.subplot(121)
    # plt.imshow(layout_seg_origin)
    # plt.subplot(122)
    # plt.imshow(trans_pred_seg)
    # plt.show()

    # return trans_pred_seg
    return trans_pred_seg.astype(np.uint8), wall_mask_seperate, split_line, ceil_, floor_


