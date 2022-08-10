import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def semantic_transfer(gt_label_list, pred_label_list, pred_seg):
    trans_pred_seg = np.zeros_like(pred_seg)

    for ite, plane_id in enumerate(pred_label_list):
        mask = pred_seg == plane_id
        trans_pred_seg[mask] = gt_label_list[ite]

    # plt.imshow(trans_pred_seg)
    # plt.show()

    return trans_pred_seg




def match_trans_by_Hungarian(gt, pred):
    # gt:(K, h, w), K represent the Gt num of layout plane
    # pred: (k, h, w), k represent the pred num of layout plane

    # plt.subplot(121)
    # plt.imshow(gt)
    # plt.subplot(122)
    # plt.imshow(pred)
    # plt.show()

    h, w = gt.shape

    # 加1的前提是标签都要从0开始
    # 天花板和地面时而有，时而没有（但是这不会对匹配结果产生影响，因为不存在的平面那个通道全是0，匹配度为0）
    # 并且假设pred_onehot的某个平面和gt_onehot第a个平面匹配度最高，那么pred_onehot的这个平面的布局语义标签就是gt_onehot中对应通道顺序（第几个通道）
    num_class_in_gt = np.max(gt)+1
    num_class_in_pred = np.max(pred)+1

    gt = torch.tensor(gt, dtype = torch.int64)
    pred = torch.tensor(pred, dtype = torch.int64)

    gt_onehot = F.one_hot(gt, num_classes = num_class_in_gt)        # (h, w, k)
    pred_onehot = F.one_hot(pred, num_classes = num_class_in_pred)

    gt_onehot = np.array(gt_onehot).reshape(-1, num_class_in_gt).T.reshape(num_class_in_gt, h, w)       #(h, w, K) --> (K, h, w)
    pred_onehot = np.array(pred_onehot).reshape(-1, num_class_in_pred).T.reshape(num_class_in_pred, h, w)   # (k, h, w)


    n = len(gt_onehot)
    m = len(pred_onehot)

    valid = (gt.sum(0) > 0).sum()   # 像素总数(作者知道下面有黑条的问题,这个是为了排除下面的黑条)
    valid = 192*256
    if m == 0:
        raise IOError
    else:
        gt_onehot = gt_onehot[:, np.newaxis, :, :]
        pred_onehot = pred_onehot[np.newaxis, :, :, :]

        # cost矩阵代表两个平面的对应程度，数值越大，代表对应两个平面匹配度越好
        # (gt_onehot+pred_onehot) == 2).shape = (K, k, h, w), K--Gt_num_plane, k--pred_num_plane
        cost = np.sum((gt_onehot + pred_onehot) == 2, axis=(2, 3))  # n*m, (K, k, h, w)-->(K, k), 统计对应平面的匹配度（重叠的像素数量）
        # print('cost matrix: \n', cost)

        # row代表开销矩阵对应的行索引， col代表对应行索引的最优指派的列索引
        # linear_sum_assignment是用来求最小开销的，所以乘以-1，将求最大匹配问题转化为求最小开销问题
        row, col = linear_sum_assignment(-1 * cost)
        # row: GT中的对应标签  col: pred中的对应标签， 对应的位置就是两个结果的对应关系
        # print('row: ', row)
        # print('col: ', col)


        inter = cost[row, col].sum()    # 通过二维索引取出对应的数值，也既最小开销值，
        # print('valid pixel: ', valid)
        # print('inter pixel: ', inter)

        PA = inter / valid      # pixel accuracy, 求出匹配像素占所有像素的比例
        # print('PA: ', PA)

        trans_pred_seg = semantic_transfer(row, col, pred)

        return PA, trans_pred_seg



