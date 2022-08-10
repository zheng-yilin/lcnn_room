import glob
import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import torch
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from lcnn.config import M

'''
假设有m根直线，n个交点（用三维坐标表示，其实三维信息没有）
字典jids中键为交点坐标，值为交点索引（索引值的得到方法见dataset/wireframe）

jmap表示交点的位置，像素内有交点的值为1，没有交点的值为0
但是交点不一定在像素的正中心，交点相对与像素中心会有uv两个方向的偏移量，通过joff表示
lmap就是将所有直线绘制出来
junc就是交点的三维坐标(n, 3)
Lpos是通过索引值来表示正样本直线，(m, 2)
Lneg是通过索引值来表示负样本直线，(4000, 2)，因为负样本直线非常多，作者只取了4000个负样本
lpos、lneg分别通过交点坐标值来表示正负样本直线

  # jmap:(1, 128, 128), 交点坐标
  # joff:(1, 2, 128, 128)，偏移量
  # lmap:(128, 128)，线图
  # junc:(100, 3)       (114, 3)
  # Lpos:(67,2)         (85, 2)
  # Lneg:(4000, 2)      (4000, 2)
  # lpos:(67, 2, 3)     (85, 2, 3)
  # lneg:(2000, 2, 3)   (2000, 2, 3)
'''

class WireframeDataset(Dataset):
    def __init__(self, rootdir, split):
        self.rootdir = rootdir
        filelist = glob.glob(f"{rootdir}/{split}/*_label.npz")      # （正则表达式，通配符）搜索对应split下所有的.npz文件
        filelist.sort()

        print(f"n{split}:", len(filelist))      # ntrain = 20000
        self.split = split
        self.filelist = filelist

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        iname = self.filelist[idx][:-10].replace("_a0", "").replace("_a1", "") + ".png"     # 对于我的数据集，这个两个replace可以不要
        # self.filelist[idx]----/home/hui/database/dataset/wireframe/wireframe/train/00060298_1_label.npz
        # 将最后十个字符除掉就剩下00060298_1
        # iname = /home/hui/database/dataset/wireframe/wireframe/train/00060298_1.png

        image = io.imread(iname).astype(float)[:, :, :3]    # (512, 512, 3)，最后不用取维度啊

        # if "a1" in self.filelist[idx]:      # 如果文件名中存在a1,就将图片进行左右翻转
        #     image = image[:, ::-1, :]   #

        image = (image - M.image.mean) / M.image.stddev     # 标准化

        image = np.rollaxis(image, 2).copy()    # (512, 512, 3)-->(3, 512, 512),

        # npz["jmap"]: [J, H, W]    Junction heat map(交点热图，像素内有交点，则这个像素的值为1 )
        # npz["joff"]: [J, 2, H, W] Junction offset within each pixel（维度2表示两个轴方向的偏移量, 像素内有交点，则这个像素就有值，否则就为0）
        # npz["lmap"]: [H, W]       Line heat map with anti-aliasing（无锯齿的直线热图,直线经过的地方有值∈(0，1]，没有直线经过的地方就为0）
        # npz["junc"]: [Na, 3]      Junction coordinates（交点的三维坐标，但是深度值全为0；交点的数量可能不等于直线数量*2，因为有些直线可能端点相交）
        # npz["Lpos"]: [Mp, 2]       Positive lines represented with junction indices（用交点索引表示的positive的线，大写的表示索引）
        # npz["Lneg"]: [Mn, 2]       Negative lines represented with junction indices（用交点索引表示的negative的线）
        # npz["lpos"]: [Np, 2, 3]   Positive lines represented with junction coordinates（用交点坐标表示的positive的线，小写的表示坐标, 第三个维度深度全是0）
        # npz["lneg"]: [Nn, 2, 3]   Negative lines represented with junction coordinates（用交点坐标表示的negative的线）
        # For junc, lpos, and lneg that stores the junction coordinates,
        # the last dimension is (y, x, t), where t represents the type of that junction.

        with np.load(self.filelist[idx]) as npz:
            target = {
                name: torch.from_numpy(npz[name]).float()
                for name in ["jmap", "joff", "lmap"]
            }   # 依次读取上面几种数据

            # print(111, target['jmap'].shape)    # (1, 128, 128)
            # print(222, target['joff'].shape)    # (1, 2, 128, 128)
            # print(333, target['lmap'].shape)    # (128, 128)

            # plt.imshow(target['lmap'])
            # plt.show()

            # static start, 帮助网络冷启动
            lpos = np.random.permutation(npz["lpos"])[: ]       # M.n_stc_posl:300, 随机排序，然后取前300个,冷启动阶段要给更多的正样本
            lneg = np.random.permutation(npz["lneg"])[: M.n_stc_negl]       # M.n_stc_negl:40，随机排序，取前40个
            npos, nneg = len(lpos), len(lneg)
            lpre = np.concatenate([lpos, lneg], 0)  # [npos+nneg, 2, 3], 等与lpos的数量加上lneg的数量

            # print(111, lpos.shape)      # (n_pos, 2, 3)，为什么后两个维度是2，3？ 因为是用两个交点的三维坐标表示直线
            # print(222, lneg.shape)      # (n_neg, 2, 3)
            # print(333, lpre.shape)      # ((n_pos+n_neg), 2, 3)
            # print(444, lpre[1])         # lpre[m]和lpre[m, ::1]是一样的，都是表示index为m的直线的两个三维端点的坐标
            # print(555, lpre[1, ::-1])   # lpre[m]和lpre[m, ::-1]表示的也是同一根直线，但是端点的顺序调换了

            for i in range(len(lpre)):
                if random.random() > 0.5:
                    lpre[i] = lpre[i, ::-1]     # 将直线两个端点的顺序调换，但表示的还是同一根直线
            ldir = lpre[:, 0, :2] - lpre[:, 1, :2]      # ((n_pos+n_neg), 2), 表示的是每根直线在xy方向上的差值（坐标为多位小数，那么肯定是xy而不是uv）

            # 将ldir的xy两个方向都单位化，也既每个方向的数值绝对值最大值都为1，但是可能为正也可能为负（或许是要转化成角度？）
            ldir /= np.clip(LA.norm(ldir, axis=1, keepdims=True), 1e-6, None)

            feat = [
                lpre[:, :, :2].reshape(-1, 4) / 128 * M.use_cood,   # use_cood = 0，作用是进行消融实验; 除以128就把坐标缩放到[0,1)之间了
                ldir * M.use_slop,      # use_slop = 0，作用是进行消融实验
                lpre[:, :, 2],      # 只取了lpre的第三个维度，也就是交点类型
            ]
            feat = np.concatenate(feat, 1)      # ((n_pos+n_neg), 8)， feat中所有元素全是0，就没啥意义啊

            # print(777, torch.from_numpy(npz["junc"][:, :2]).shape)
            # print(888, torch.from_numpy(npz["junc"][:, 2]).byte())
            # print(999, torch.cat([torch.ones(npos), torch.zeros(nneg)]))
            # print(10111, self.adjacency_matrix(len(npz["junc"]), npz["Lpos"]).shape)
            # print(10222, self.adjacency_matrix(len(npz["junc"]), npz["Lneg"]).shape)

            # plt.imshow(self.adjacency_matrix(len(npz["junc"]), npz["Lpos"]))
            # plt.show()

            meta = {
                "junc": torch.from_numpy(npz["junc"][:, :2]),   # (Na, 2)交点的xy坐标（Na为交点数量，交点的数量可能不等于直线数量*2，因为有些直线可能端点相交）
                "jtyp": torch.from_numpy(npz["junc"][:, 2]).byte(),      # (Na), 表示交点类型，全都是0
                "Lpos": self.adjacency_matrix(len(npz["junc"]), npz["Lpos"]),       # (Na+1, Na+1), 用连接矩阵表示的直线，值为1处有直线
                "Lneg": self.adjacency_matrix(len(npz["junc"]), npz["Lneg"]),       # (Na+1, Na+1)
                "lpre": torch.from_numpy(lpre[:, :, :2]),   # [n_pos + n_neg, 2, 2], 只取了前两个维度，没要第三个深度值。所有直线的两个端点的uv坐标
                "lpre_label": torch.cat([torch.ones(npos), torch.zeros(nneg)]),     # 一个一维tensor，前面是npos个1.，后面全都是0,这是直线的标签（是否为正负样本）
                "lpre_feat": torch.from_numpy(feat),
            }

            # junc = torch.from_numpy(npz["junc"][:, :2])
            # jtyp = torch.from_numpy(npz["junc"][:, 2]).byte()
            # Lpos = self.adjacency_matrix(len(npz["junc"]), npz["Lpos"])
            # Lneg = self.adjacency_matrix(len(npz["junc"]), npz["Lneg"])
            # lpre = torch.from_numpy(lpre[:, :, :2])
            # lpre_label = torch.cat([torch.ones(npos), torch.zeros(nneg)])
            # lpre_feat = torch.from_numpy(feat)
            #
            # print('@'*50)
            # print(junc.shape)
            # print(jtyp.shape)
            # print(Lpos.shape)
            # print(Lneg.shape)
            # print(lpre.shape)
            # print(lpre_label.shape)
            # print(lpre_feat.shape)


        # return torch.from_numpy(image).float(), junc, jtyp, Lpos, Lneg, lpre, lpre_label, lpre_feat, target

        # target中包含["jmap", "joff", "lmap"]
        return torch.from_numpy(image).float(), meta, target

    def adjacency_matrix(self, n, link):
        # 为什么要加1呢？ 为了line_vectorizer.py中排除明显不正确的点
        mat = torch.zeros(n + 1, n + 1, dtype=torch.uint8)
        link = torch.from_numpy(link)
        if len(link) > 0:
            mat[link[:, 0], link[:, 1]] = 1
            mat[link[:, 1], link[:, 0]] = 1
        return mat


def collate(batch):
    # batch是一个列表，其内包含tuple，tuple数量等于bs，每个tuple内就包含了一组数据


    return (
        default_collate([b[0] for b in batch]),
        [b[1] for b in batch],
        default_collate([b[2] for b in batch]),
    )
