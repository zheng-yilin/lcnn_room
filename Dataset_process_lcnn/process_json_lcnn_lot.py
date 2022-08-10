#!/usr/bin/env python
"""Process Huang's wireframe dataset for L-CNN network
Usage:
    dataset/wireframe.py <src> <dst>
    dataset/wireframe.py (-h | --help )

Examples:
    python dataset/wireframe.py /datadir/wireframe data/wireframe

Arguments:
    <src>                Original data directory of Huang's wireframe dataset
    <dst>                Directory of the output

Options:
   -h --help             Show this screen.
"""

import os
import sys
import json
from itertools import combinations

import cv2
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from docopt import docopt
from scipy.ndimage import zoom

try:
    sys.path.append("")
    sys.path.append("..")
    from lcnn.utils import parmap
except Exception:
    raise


def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))


def save_heatmap(prefix, image, lines):
    im_rescale = (512, 512)
    heatmap_scale = (128, 128)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]   # 缩放比例
    jmap = np.zeros((1,) + heatmap_scale, dtype=np.float32)     # (1, 128, 128)
    joff = np.zeros((1, 2) + heatmap_scale, dtype=np.float32)   # (1, 2, 128, 128)
    lmap = np.zeros(heatmap_scale, dtype=np.float32)    # (128, 128)

    # 图片缩放了，交点的坐标也要相应变换
    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)
    lines = lines[:, :, ::-1]   # 将交点顺序改变，有啥用？

    junc = []
    jids = {}

    def jid(jun):  # input:[48.21580208 41.28555078], output:43
        jun = tuple(jun[:2])
        if jun in jids:
            return jids[jun]
        jids[jun] = len(junc)   # jids以键值对的形式保存交点索引（键是元祖形式的交点坐标）
        junc.append(np.array(jun + (0,)))   # 给交点加上z值，不过全是0
        return len(junc) - 1    # 返回交点索引

    lnid = []
    lpos, lneg = [], []
    # print('#'*100)
    for v0, v1 in lines:    # v0(u0, v0), v1(u1, v1)代表了直线两个端点的坐标
        # print(v0)
        # print('*'*50, jid(v0))
        lnid.append((jid(v0), jid(v1)))     # 以交点索引的形式保存直线
        lpos.append([junc[jid(v0)], junc[jid(v1)]])     # 以两个三维交点的形式保存直线

        vint0, vint1 = to_int(v0), to_int(v1)   # 交点坐标，向下取整
        jmap[0][vint0] = 1  # 将有交点的像素值变为0
        jmap[0][vint1] = 1
        rr, cc, value = skimage.draw.line_aa(*to_int(v0), *to_int(v1))
        # print('*'*50)
        # print(len(value))
        # print(value)

        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)  # 在lmap上绘制直线，像素的值属于(0,1]

    for v in junc:
        vint = to_int(v[:2])
        joff[0, :, vint[0], vint[1]] = v[:2] - vint - 0.5   # 交点uv坐标-整数部分-0.5——交点距离像素中心的距离（量纲是像素）

    llmap = zoom(lmap, [0.5, 0.5])      # 将lmap缩小为原来的1/2，但是值可能会超过1
    # plt.subplot(121)
    # plt.imshow(lmap)
    # plt.subplot(122)
    # plt.imshow(llmap)
    # plt.show()

    lineset = set([frozenset(l) for l in lnid])   # 一个冻结的集合，冻结后集合不能再添加或删除任何元素。
    # print('*'*50)
    # print(lnid)
    # print(lineset)
    # itertools.combinations(iterable, r):创建一个迭代器，返回iterable中所有长度为r的子序列，返回的子序列中的项按输入iterable中的顺序排序
    # 就是将所有Gt交点两两配对，其中包括了Pos line 和Neg line
    for i0, i1 in combinations(range(len(junc)), 2):
        if frozenset([i0, i1]) not in lineset:  # 排除掉正样本
            v0, v1 = junc[i0], junc[i1]
            vint0, vint1 = to_int(v0[:2] / 2), to_int(v1[:2] / 2)
            rr, cc, value = skimage.draw.line_aa(*vint0, *vint1)
            lneg.append([v0, v1, i0, i1, np.average(np.minimum(value, llmap[rr, cc]))]) # 前两个是uv坐标，后两个是junc中的index

    assert len(lneg) != 0
    lneg.sort(key=lambda l: -l[-1])

    junc = np.array(junc, dtype=np.float32)
    Lpos = np.array(lnid, dtype=np.int)
    Lneg = np.array([l[2:4] for l in lneg][:4000], dtype=np.int)
    lpos = np.array(lpos, dtype=np.float32)
    lneg = np.array([l[:2] for l in lneg[:2000]], dtype=np.float32)

    image = cv2.resize(image, im_rescale)

    # plt.subplot(131), plt.imshow(lmap)
    # plt.subplot(132), plt.imshow(image)
    # for i0, i1 in Lpos:
    #     plt.scatter(junc[i0][1] * 4, junc[i0][0] * 4)
    #     plt.scatter(junc[i1][1] * 4, junc[i1][0] * 4)
    #     plt.plot([junc[i0][1] * 4, junc[i1][1] * 4], [junc[i0][0] * 4, junc[i1][0] * 4])
    # plt.subplot(133), plt.imshow(lmap)
    # for i0, i1 in Lneg[:150]:
    #     plt.plot([junc[i0][1], junc[i1][1]], [junc[i0][0], junc[i1][0]])
    # plt.show()

    # For junc, lpos, and lneg that stores the junction coordinates, the last
    # dimension is (y, x, t), where t represents the type of that junction.  In
    # the wireframe dataset, t is always zero.
    np.savez_compressed(
        f"{prefix}_label.npz",
        aspect_ratio=image.shape[1] / image.shape[0],
        jmap=jmap,  # [J, H, W]    Junction heat map
        joff=joff,  # [J, 2, H, W] Junction offset within each pixel
        lmap=lmap,  # [H, W]       Line heat map with anti-aliasing
        junc=junc,  # [Na, 3]      Junction coordinate
        Lpos=Lpos,  # [M, 2]       Positive lines represented with junction indices
        Lneg=Lneg,  # [M, 2]       Negative lines represented with junction indices
        lpos=lpos,  # [Np, 2, 3]   Positive lines represented with junction coordinates
        lneg=lneg,  # [Nn, 2, 3]   Negative lines represented with junction coordinates
    )
    cv2.imwrite(f"{prefix}.png", image)

    # plt.imshow(jmap[0])
    # plt.savefig("/tmp/1jmap0.jpg")
    # plt.imshow(jmap[1])
    # plt.savefig("/tmp/2jmap1.jpg")
    # plt.imshow(lmap)
    # plt.savefig("/tmp/3lmap.jpg")
    # plt.imshow(Lmap[2])
    # plt.savefig("/tmp/4ymap.jpg")
    # plt.imshow(jwgt[0])
    # plt.savefig("/tmp/5jwgt.jpg")
    # plt.cla()
    # plt.imshow(jmap[0])
    # for i in range(8):
    #     plt.quiver(
    #         8 * jmap[0] * cdir[i] * np.cos(2 * math.pi / 16 * i),
    #         8 * jmap[0] * cdir[i] * np.sin(2 * math.pi / 16 * i),
    #         units="xy",
    #         angles="xy",
    #         scale_units="xy",
    #         scale=1,
    #         minlength=0.01,
    #         width=0.1,
    #         zorder=10,
    #         color="w",
    #     )
    # plt.savefig("/tmp/6cdir.jpg")
    # plt.cla()
    # plt.imshow(lmap)
    # plt.quiver(
    #     2 * lmap * np.cos(ldir),
    #     2 * lmap * np.sin(ldir),
    #     units="xy",
    #     angles="xy",
    #     scale_units="xy",
    #     scale=1,
    #     minlength=0.01,
    #     width=0.1,
    #     zorder=10,
    #     color="w",
    # )
    # plt.savefig("/tmp/7ldir.jpg")
    # plt.cla()
    # plt.imshow(jmap[1])
    # plt.quiver(
    #     8 * jmap[1] * np.cos(tdir),
    #     8 * jmap[1] * np.sin(tdir),
    #     units="xy",
    #     angles="xy",
    #     scale_units="xy",
    #     scale=1,
    #     minlength=0.01,
    #     width=0.1,
    #     zorder=10,
    #     color="w",
    # )
    # plt.savefig("/tmp/8tdir.jpg")


def main():
    # args = docopt(__doc__)
    # data_root = args["<src>"]
    # data_output = args["<dst>"]

    data_root = '/home/hui/database/dataset/interiornet_layout/'
    # data_root = '/home/hui/database/dataset/LSUN/'

    # for batch in ["training", "validation"]:
    for batch in ["train"]:
    # for batch in ["test"]:
    # for batch in ["val"]:
    # for batch in ["training"]:
    # for batch in ["validation"]:
        # data_output = os.path.join(data_root, 'il_lcnn')
        data_output = os.path.join(data_root, 'LSUN_lcnn')
        os.makedirs(data_output, exist_ok=True)

        sample_range = [4000, 6534]
        threshold = 0.95
        anno_file = os.path.join(data_root, batch, f"{batch}_{sample_range[0]}_{sample_range[1]}_{threshold}.json")    # 标注信息

        with open(anno_file, "r") as f:
            dataset = json.load(f)  # 对于每张图片，包括图片名、直线交点uv坐标（四个交点表示一根直线）、图片高、图片宽
            # print(dataset)

        def handle(data):
            im = cv2.imread(os.path.join(data_root, batch, "image", data["filename"]))      # il
            # im = cv2.imread(os.path.join(data_root, batch, "images", data["filename"]))     # LSUN
            print(os.path.join(data_root, "image", data["filename"]))

            prefix = data["filename"].split(".")[0]
            lines = np.array(data["lines"]).reshape(-1, 2, 2)
            os.makedirs(os.path.join(data_output, batch), exist_ok=True)

            lines0 = lines.copy()

            # 下面这些是为了数据增广
            lines1 = lines.copy()

            # lines0(1, 2, ..).shape = (N, 2, 2), lines1[:, :, 0].shape = (N, 2), N代表的是交点数量
            lines1[:, :, 0] = im.shape[1] - lines1[:, :, 0]     # 将每个交点的u坐标替换为[图片宽-每个交点的u坐标]
            lines2 = lines.copy()
            lines2[:, :, 1] = im.shape[0] - lines2[:, :, 1]     # 将每个交点的u坐标替换为[图片高-每个交点的v坐标]
            lines3 = lines.copy()
            lines3[:, :, 0] = im.shape[1] - lines3[:, :, 0]     # 替换u
            lines3[:, :, 1] = im.shape[0] - lines3[:, :, 1]     # 替换v

            path = os.path.join(data_output, batch, prefix)
            save_heatmap(f"{path}_0", im[::, ::], lines0)
            # if batch == "training":
            if batch == "train":
                save_heatmap(f"{path}_1", im[::, ::-1], lines1)
                save_heatmap(f"{path}_2", im[::-1, ::], lines2)
                save_heatmap(f"{path}_3", im[::-1, ::-1], lines3)
            print("Finishing", os.path.join(data_output, batch, prefix))

        parmap(handle, dataset, 16)


if __name__ == "__main__":
    main()
