import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import cv2


# 创建一个字典来保存没个墙面标签搜索到的多变形，并从中选出像素数量最多的那个多边形
class PolygonCandidate:
    def __init__(self, layout_seg, dataset):
        self.h, self.w = layout_seg.shape
        self.layout_seg = layout_seg
        self.wall_in_dataset = {'Lsun': [1, 2, 3], 'InteriorNet-Layout': [4, 5, 6, 7, 8],
                                'Matterport3D-Layout': [4, 5, 6, 7, 8], 'NYU_v2_303': [4, 5, 6, 7, 8]}
        self.cf_in_dataset = {'Lsun': [4, 5], 'InteriorNet-Layout': [1, 3],
                              'Matterport3D-Layout': [1, 2, 3], 'NYU_v2_303': [1, 2, 3]}
        self.label_of_cf = self.cf_in_dataset[dataset]
        self.label_of_wall = self.wall_in_dataset[dataset]

        self.candidate, self. k_list_ex_cf = self.generate_candidate(layout_seg)

    def generate_candidate(self, layout_seg):
        k_v_dict = Counter(layout_seg.reshape(-1))
        k_array = np.array(list(k_v_dict.keys()))
        v_array = np.array(list(k_v_dict.values()))
        # print(k_array)
        # print(v_array)

        sum_pixels = np.sum(v_array)
        ratio_array = v_array/sum_pixels # LSUN给的layout seg有些问题，莫名其妙多出不存在的标签，则以比例为判断标准而不是像素数量
        # print(ratio_array)
        conform_index = ratio_array > 0.002
        k_array_ = list(k_array[conform_index])

        k_list_ex_cf = []
        for i in k_array_:
            if i in self.label_of_cf:
                continue
            k_list_ex_cf.append(i)

        candidate = {}
        for i in k_list_ex_cf:
            candidate[i] = {'num_pixel': [], 'plane_points': []}
        return candidate, k_list_ex_cf

    def update(self, wall_label, num_pixel, plane_points):
        self.candidate[wall_label]['num_pixel'].append(num_pixel)
        self.candidate[wall_label]['plane_points'].append(plane_points)

    def determine_final_polygon(self, float_points_origin, int_points_origin, int_points):
        k_list = self.k_list_ex_cf
        # print('k_list:', k_list)

        # 缩放和未缩放的对应关系
        match_dict_scale = self.match_two_list(int_points, int_points_origin)
        # 整数和小数的对应关系
        match_dict_int_float = self.match_two_list(int_points_origin, float_points_origin)

        self.final_polygons_int = {}
        self.final_polygons_int_scale = {}
        self.final_polygons_float = {}

        for label_ in k_list:   # 遍历每个墙面标签
            one_wall_candis = self.candidate[label_]    # 取出某个墙面的数据
            num_pixel_one_wall = one_wall_candis['num_pixel']   # 待选多边形的像素数量
            plane_points_one_wall = one_wall_candis['plane_points'] # 待选多边形的顶点（顺时针排序）

            # 如果某墙面标签对应的多边形数量为0，说明阈值设得大了，单独处理
            if len(num_pixel_one_wall) == 0:
                return None, None

            max_pixel_one_wall = max(num_pixel_one_wall)    # 所有多边形中最大的像素数量
            index_max = num_pixel_one_wall.index(max_pixel_one_wall)    # 最大像素数量多边形的索引
            plane_points_max_ = plane_points_one_wall[index_max]     # 最大像素数量多边形的顶点

            self.final_polygons_int_scale[label_] = plane_points_max_

            # 这里将可能缩放了的替换成未缩放的，所以需要 缩放-未缩放的匹配字典
            plane_points_max = []
            for vertex_ in plane_points_max_:
                plane_points_max.append(match_dict_scale[tuple(vertex_)])

            self.final_polygons_int[label_] = plane_points_max


            # tran the point from int to float one
            plane_point_float = []
            for point_ in plane_points_max:
                float_one_ = match_dict_int_float[tuple(point_)]
                plane_point_float.append(float_one_)

            self.final_polygons_float[label_] = np.array(plane_point_float, np.float64)

        return self.final_polygons_int, self.final_polygons_float

    def poly_visualize(self):
        polygons = self.final_polygons_int_scale.values()
        wall_mask_inte = np.zeros((self.h, self.w), np.uint8)

        wall_masks_sepe = []
        for poly_ in polygons:
            canvas = np.zeros((self.h, self.w), np.uint8)
            mask_ = cv2.fillConvexPoly(canvas, poly_, 1)

            wall_mask_inte += mask_
            wall_masks_sepe.append(mask_)

        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.imshow(self.layout_seg)
        plt.subplot(122)
        plt.imshow(wall_mask_inte)
        plt.show()

        # return wall_mask_inte, wall_masks_sepe

    def generate_graph_OnlyKeypoints(self):
            pass

    def generate_graph_CloseLoop(self):
        label_wall = self.final_polygons_float.keys()
        vertex_wall = self.final_polygons_float.values()    # 图片中包含的墙面的顶点坐标
        # print(label_wall)
        # print(vertex_wall)

        for poly_index_, poly_vertex_ in enumerate(vertex_wall):
            # label_poly = list(label_wall)[poly_index_]
            num_vertex = len(poly_vertex_)

            ep1_index = np.arange(num_vertex)
            ep2_index = np.append(ep1_index[1:], ep1_index[0])

            ep1 = poly_vertex_[ep1_index]
            ep2 = poly_vertex_[ep2_index]

            line_of_polygon = np.concatenate([ep1, ep2], axis=1)

            if poly_index_ == 0:
                lines_repeat = line_of_polygon
                # 当墙面数量为1时，是没有重复直线的，直接返回直线即可
                if len(vertex_wall) == 1:
                    return lines_repeat
            else:
                lines_repeat = np.concatenate([lines_repeat, line_of_polygon], axis=0)

        # 对于两个墙面相交的直线，两个多边形中都包含，那么只能从一个多边形中选出一根
        line_repeat_reverse_1 = lines_repeat[:, 0:2]
        line_repeat_reverse_2 = lines_repeat[:, 2: ]
        line_repeat_reverse = np.concatenate([line_repeat_reverse_2, line_repeat_reverse_1], axis=1)    # switch the order of ep1, 2
        # print(lines_repeat)
        # print('~'*50)
        # print(line_repeat_reverse)
        # print('#'*50)

        repeat_lines = []
        non_repeat_lines = []

        for line_ in lines_repeat:
            # 返回一维数组在二维数组中的下标（可能二维数组中包含多个该一维数组），如果不存在，则会返回[]，如果存在，则会返回该一维数组在二维数组中的下标[n1, n2, ..]
            subindex = np.where((line_repeat_reverse==line_).all(1))[0]
            if len(subindex) != 0:
                repeat_lines.append(line_)
                continue
            non_repeat_lines.append(line_)

        repeat_lines = np.array(repeat_lines)
        non_repeat_lines = np.array(non_repeat_lines)

        num_repeat_lines_fact = len(repeat_lines)

        num_wall = len(self.k_list_ex_cf)
        num_repeat_lines_theory = (num_wall - 1) * 2

        # 如果实际的重复直线数量与理论的交线数量不等，那么说明阈值设小了，需要单独处理(这里就包括了repeat_lines = []的情况)
        if num_repeat_lines_fact != num_repeat_lines_theory:
            return []

        # print(77777777)
        # print(repeat_lines)

        sum_repeat = np.sum(repeat_lines, axis=1)
        sum_repeat = np.round(sum_repeat, 3)
        unique_lines = np.array(list(set(sum_repeat)))

        repeat_line_group = []
        for unique_ in unique_lines:
            index_ = np.where((sum_repeat == unique_))[0]

            repeat_line_group.append(index_)


        repeat_line_group = np.array(repeat_line_group)
        # print(888888888)
        # print(repeat_line_group)

        index_ride_repeat = repeat_line_group[:, 0]

        # print(999)
        # print(index_ride_repeat)
        line_ride_repeat = repeat_lines[index_ride_repeat]

        final_graph = np.concatenate([non_repeat_lines, line_ride_repeat], axis=0)

        return final_graph


    def match_two_list(self, source_list, target_list):
        match_dict = {}

        for index, point_ in enumerate(source_list):
            match_dict[tuple(point_)] = target_list[index]

        return match_dict









