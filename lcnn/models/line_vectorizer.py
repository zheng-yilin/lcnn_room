import itertools
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lcnn.config import M

FEATURE_DIM = 8


class LineVectorizer(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        lambda_ = torch.linspace(0, 1, M.n_pts0)[:, None]   # n_pts0=32
        self.register_buffer("lambda_", lambda_)
        self.do_static_sampling = M.n_stc_posl + M.n_stc_negl > 0   # M.n_stc_posl = 300, M.n_stc_negl = 40

        self.fc1 = nn.Conv2d(256, M.dim_loi, 1)     # dim_loi = 128
        scale_factor = M.n_pts0 // M.n_pts1     # n_pts1 = 8, 所以scale_factor = 4, 特征图缩小倍数
        if M.use_conv:  # 默认0，使用一维卷积Bottleneck1D，或者只使用线性层
            self.pooling = nn.Sequential(
                nn.MaxPool1d(scale_factor, scale_factor),
                Bottleneck1D(M.dim_loi, M.dim_loi),
            )
            self.fc2 = nn.Sequential(
                nn.ReLU(inplace=True), nn.Linear(M.dim_loi * M.n_pts1 + FEATURE_DIM, 1)
            )
        else:
            self.pooling = nn.MaxPool1d(scale_factor, scale_factor)
            self.fc2 = nn.Sequential(
                nn.Linear(M.dim_loi * M.n_pts1 + FEATURE_DIM, M.dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(M.dim_fc, M.dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(M.dim_fc, 1),
            )
        self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input_dict):
        '''
        input_dict = {
                     "image": recursive_to(image, self.device),      # recursive_to：将数据放到gpu上
                     "meta": recursive_to(meta, self.device),
                     "target": recursive_to(target, self.device),
                     "mode": "training",
                     }
         meta = {
                 "junc": torch.from_numpy(npz["junc"][:, :2]),   # (Na, 2)交点的xy坐标（Na为交点数量，交点的数量可能不等于直线数量*2，因为有些直线可能端点相交）
                 "jtyp": torch.from_numpy(npz["junc"][:, 2]).byte(),      # (Na), 好像全都是0（还不知道是干嘛的）
                 "Lpos": self.adjacency_matrix(len(npz["junc"]), npz["Lpos"]),       # (Na+1, Na+1),连接矩阵是个对称矩阵，相连为1，不相连为0,(Na+1怎么来的？)
                 "Lneg": self.adjacency_matrix(len(npz["junc"]), npz["Lneg"]),       # (Na+1, Na+1)
                 "lpre": torch.from_numpy(lpre[:, :, :2]),   # [n_pos+ n_neg, 2, 3], 直线的三维交点在xy方向上的归一化后的距离
                 "lpre_label": torch.cat([torch.ones(npos), torch.zeros(nneg)]),     # 一个一维tensor，前面是npos个1.，后面全都是0（这是干嘛用的）
                 "lpre_feat": torch.from_numpy(feat),
                }

        result = {
                # 第二个fc层的输出
                "feature": feature,

                # 最后一个hg module的输出（网络最后的输出）
                'pred's: {
                         'jamp': ~,             # (b, j, h, w)
                         'lmap': ~,             # (b, h, w)
                         'joff': ~}             # (b, j, c, h, w)

                # 两个hg module的损失
                'loss': {
                          [OrderedDict([('jmap', tensor([0.6529, 0.6508], device='cuda:0', grad_fn=<MulBackward0>)),
                                        ('lmap', tensor([0.1423, 0.1652], device='cuda:0', grad_fn=<MulBackward0>)),
                                        ('joff', tensor([0.1230, 0.1317], device='cuda:0', grad_fn=<MulBackward0>))]),
                           OrderedDict([('jmap', tensor([0.8486, 0.8866], device='cuda:0', grad_fn=<MulBackward0>)),
                                        ('lmap', tensor([0.1496, 0.1739], device='cuda:0', grad_fn=<MulBackward0>)),
                                        ('joff', tensor([0.1259, 0.1295], device='cuda:0', grad_fn=<MulBackward0>))])]
                              }
                }
        '''

        # 这里的backbone实际上指的是 主干网络+[junction proposal module]
        result = self.backbone(input_dict)

        h = result["preds"]     # 第二个沙漏网络的fc层输出
        # print('before fc1', result["feature"].shape)
        x = self.fc1(result["feature"])     # 256-->128通道
        # print('after fc1', x.shape)
        n_batch, n_channel, row, col = x.shape

        # xs保存多张图片中的直线经过LoI提取到的特征，ys保存多张图片中的直线的标签；fs不管
        # ps保存多张图片中的直线的端点uv坐标；idx表示一个batch中每张图片的图片的累计量，比如[0, a, a+b, a+b+c]; jcs保存每张图片中概率大于0.03的交点
        xs, ys, fs, ps, idx, jcs = [], [], [], [], [0], []
        # input_dict["meta"] = [meta1, meta1, ...]，其中每个bs_meta都是dataset中的字典
        # input_dict["meta"]中的元素数量等于batch_size
        for i, meta in enumerate(input_dict["meta"]):   # 对每个batch进行循环

            # print(111, h['jmap'].shape)
            # print(222, h['joff'].shape)
            # print(333, h['lmap'].shape)

            # 这个是动态采样器
            # p.shape = [选出的直线数量， 2， 2], label.shape = [选出的直线数量], feat, jcs.shape = [socre>0.03的点数，2]
            p, label, feat, jc = self.sample_lines( meta, h["jmap"][i], h["joff"][i], input_dict["mode"])   # h["jmap"][i],这里的[i]作用是取出某个batch，进行逐batch的运算
            # print("p.shape:", p.shape)
            ys.append(label)

            if input_dict["mode"] == "training" and self.do_static_sampling:    # 这里是静态直线采样器
                # sample_lines函数的输出p 表示的是由预测交点
                p = torch.cat([p, meta["lpre"]])
                feat = torch.cat([feat, meta["lpre_feat"]])
                ys.append(meta["lpre_label"])
                del jc
            else:
                jcs.append(jc)
                ps.append(p)
            fs.append(feat)

            #self.lambaa_.shape = [32, 1]，里面的数值就是对[0, 1]进行32等分得到的, 32表示对一根直线等距采样32个点，方便后面的LoI pooling提取特征
            # [766, 1, 2] * [32, 1] = [766, 32, 2]
            # print(1111111111)
            # print(p.shape)
            p = p[:, 0:1, :] * self.lambda_ + p[:, 1:2, :] * (1 - self.lambda_) - 0.5
            # print(p.shape)
            p = p.reshape(-1, 2)  # [N_LINE x N_POINT, 2_XY], 原本用两个端点表示一根直线，现在用在直线上采样的32个点表示一根直线
            px, py = p[:, 0].contiguous(), p[:, 1].contiguous()     # 所有点的x, y坐标， px(py).shape = [N_LINE x N_POINT]
            px0 = px.floor().clamp(min=0, max=127)  # 对采样点的坐标向下取整，转化为uv坐标（前面的坐标是小数形式的）
            py0 = py.floor().clamp(min=0, max=127)
            px1 = (px0 + 1).clamp(min=0, max=127)
            py1 = (py0 + 1).clamp(min=0, max=127)
            px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

            # LoI提取特征，下面这一部分是双线性插值
            # xp: [N_LINE, N_CHANNEL, N_POINT]  # 按论文中的：[N_LINE, 128, 32]
            xp = (
                (
                    x[i, :, px0l, py0l] * (px1 - px) * (py1 - py)
                    + x[i, :, px1l, py0l] * (px - px0) * (py1 - py)
                    + x[i, :, px0l, py1l] * (px1 - px) * (py - py0)
                    + x[i, :, px1l, py1l] * (px - px0) * (py - py0)
                )
                .reshape(n_channel, -1, M.n_pts0)
                .permute(1, 0, 2)
            )

            # 对提取到的特征, 根据scale factor进行维度缩放，[N_LINE, 128, 32]-->[N_LINE, 128, 8]
            # print('bf plooling', xp.shape)
            xp = self.pooling(xp)   # xp中保存的是某张图片经过LoI提取到的特征
            # print('af plooling', xp.shape)
            # print(11111, xp.shape)
            xs.append(xp)   # xs中保存的是一个batch的图片提取到的特征
            idx.append(idx[-1] + xp.shape[0])
            # idx保存的是累加的图片的特征点数量
            # 比如一个btach中有两张图片，第一张图片中经过LoI提取到了[a, 128, 8]的特征，第二张图片提取到了[b, 128, 8]的特征
            # 则idx由[0]-->[0, a]-->[0, a, a+b]

        x, y = torch.cat(xs), torch.cat(ys)  # ys表示的是所有点的标签，shape的理解和xs相同
        # print(22222)
        # print(x.shape)
        # print(y.shape)
        f = torch.cat(fs)
        x = x.reshape(-1, M.n_pts1 * M.dim_loi)     # [N_LINE, 128, 8]-->[N_LINE, 128*8]

        # f is represents hardcoded feature. By default, it is just a zero vector.
        x = torch.cat([x, f], 1)    # cat([N_LINE, 1024], [N_LINE, 8]) = [N_LINE, 1032]
        # print('bf fc2', x.shape)
        x = self.fc2(x).flatten()   # [N_LINE, 1032]-->[N_LINE]
        # print('af fc2', x.shape)

        # test走这个循环
        if input_dict["mode"] != "training":
            p = torch.cat(ps)   # 预测的和真实的直线
            s = torch.sigmoid(x)
            b = s > 0.5
            lines = []
            score = []
            for i in range(n_batch):
                p0 = p[idx[i] : idx[i + 1]]
                s0 = s[idx[i] : idx[i + 1]]
                mask = b[idx[i] : idx[i + 1]]
                p0 = p0[mask]
                s0 = s0[mask]
                if len(p0) == 0:    # 概率全小于0.5， 则用无意义的数据填充
                    lines.append(torch.zeros([1, M.n_out_line, 2, 2], device=p.device))     # M.n_out_line = 2500
                    score.append(torch.zeros([1, M.n_out_line], device=p.device))
                else:   # 如果有直线的概率大于0.5，则取出这些直线
                    arg = torch.argsort(s0, descending=True)    # 根据评分排序，返回顺序
                    p0, s0 = p0[arg], s0[arg]
                    lines.append(p0[None, torch.arange(M.n_out_line) % len(p0)])
                    score.append(s0[None, torch.arange(M.n_out_line) % len(s0)])
                for j in range(len(jcs[i])):
                    if len(jcs[i][j]) == 0:
                        jcs[i][j] = torch.zeros([M.n_out_junc, 2], device=p.device)
                    jcs[i][j] = jcs[i][j][
                        None, torch.arange(M.n_out_junc) % len(jcs[i][j])
                    ]   # M.n_out_junc = 250
            result["preds"]["lines"] = torch.cat(lines)
            result["preds"]["score"] = torch.cat(score)
            result["preds"]["juncs"] = torch.cat([jcs[i][0] for i in range(n_batch)])
            if len(jcs[i]) > 1:
                result["preds"]["junts"] = torch.cat(
                    [jcs[i][1] for i in range(n_batch)]
                )

        # train走这个循环
        if input_dict["mode"] != "testing":
            y = torch.cat(ys)
            # print(333333)
            # print(x.shape)
            # print(y.shape)

            loss = self.loss(x, y)
            lpos_mask, lneg_mask = y, 1 - y
            loss_lpos, loss_lneg = loss * lpos_mask, loss * lneg_mask

            def sum_batch(x):
                xs = [x[idx[i] : idx[i + 1]].sum()[None] for i in range(n_batch)]
                return torch.cat(xs)

            lpos = sum_batch(loss_lpos) / sum_batch(lpos_mask).clamp(min=1)
            lneg = sum_batch(loss_lneg) / sum_batch(lneg_mask).clamp(min=1)

            # 只对最后一个hg模块的输出求lpos及lneg损失
            result["losses"][0]["lpos"] = lpos * M.loss_weight["lpos"]
            result["losses"][0]["lneg"] = lneg * M.loss_weight["lneg"]

        if input_dict["mode"] == "training":
            del result["preds"]

        return result

    def sample_lines(self, meta, jmap, joff, mode): #   jmap.shape = (n_jtyp, h, w), joff.shape = (n_jtyp, c(2), h, w)
        # meta是GT信息，jmap，joff是预测的

        with torch.no_grad():
            junc = meta["junc"]  # [N, 2]，交点的xy坐标，小数
            jtyp = meta["jtyp"]  # [N]，全是0
            Lpos = meta["Lpos"]  # [N+1, N+1], Positive lines represented with junction indices（用连接矩阵表示的positive的线，大写的表示索引）
            Lneg = meta["Lneg"]  # [N+1, N+1], Negative lines represented with junction indices（用连接矩阵表示的negative的线）

            n_type = jmap.shape[0]  # 1
            jmap = non_maximum_suppression(jmap).reshape(n_type, -1)    # 进行非极大值抑制
            joff = joff.reshape(n_type, 2, -1)  # (1, 2, h*w)
            max_K = M.n_dyn_junc // n_type  # n_dyn_junc=300
            N = len(junc)   # N: GT交点数量

            if mode != "training":
                K = min(int((jmap > M.eval_junc_thres).float().sum().item()), max_K)    # K取预测的jmap中的点数与max_K的最大值(基本上是小于300的)
            else:
                K = min(int(N * 2 + 2), max_K)  # 训练模式下K的数值
            if K < 2:
                K = 2
            device = jmap.device

            # index: [N_TYPE, K]
            # torch.topk(): 沿着指定维度返回输入张量的k个最大值，如果不指定维度，则默认为输入张量的最后一维
            # 返回一个元组 (values,indices)，其中indices是原始输入张量input中测元素下标

            # score(index).shape = [1, n`]，n`表示经过NMS后的jmap上剩余的交点数量。
            # 如果在训练模式中，n`= min(int(N * 2 + 2), max_K)
            score, index = torch.topk(jmap, k=K)    # index表示把jmap展平后的点的索引

            # torch.gather: 获取输入向量的指定维度的指定数值
            # pytorch中tensor的第一个维度是行，index整除128，可以得到交点的行索引；index对128取余，可以得到交点的列索引，也就是x坐标对于远点的偏移量
            # 一个交点的确切位置由其所处像素的行列索引，再加上偏移量决定，而偏移量是相对于bin的中心计算的，所以需要加上个0.5，从[-1/2, 1/2]-->[0, 1]
            y = (index // 128).float() + torch.gather(joff[:, 0], 1, index) + 0.5   # x(y).shape = (1, n`)
            x = (index % 128).float() + torch.gather(joff[:, 1], 1, index) + 0.5

            # xy: [N_TYPE, K, 2], 数组索引中的None表示加一个维度
            xy = torch.cat([y[..., None], x[..., None]], dim=-1)    # xy.shape = [1, n`, 2]将一个点的x，y坐标进行堆叠
            xy_ = xy[..., None, :]
            del x, y, index

            # dist: [N_TYPE, K, N]
            '''这里类似于PlaneAE中点对应关系的算法
            将预测的n`个点和Gt的N个点两两求距离，则会得到一个[n`, N]的矩阵dist
            基于dist，在点的通道上求最小距离，得到cost与match， shape均为[1, n`]
            cost表示两个点之间的距离数值，match表示预测点与GT点的配对关系
            match = [32, 16, ...]，数值32表示预测的索引为0的点与Gt中索引为32的点向配对。
            
            需要注意的是，match中的数值表示GT点的索引，所以max(match) <= len(junc-1)
            一个Gt点可能和对个预测交点对应，但是一个预测交点只会和一个GT点匹配
            '''
            dist = torch.sum((xy_ - junc) ** 2, -1) # l2损失, dist.shape = [1, n`, N]， 其中n`为预测的交点，N为真实的交点
            cost, match = torch.min(dist, -1)       # cost(match).shape = [1, n`]

            # xy: [N_TYPE * K, 2]
            # match: [N_TYPE, K]
            for t in range(n_type):     # n_type = 1， 这个for循环啥都没做
                match[t, jtyp[match[t]] != t] = N
                # t = 0
                # aaa = jtyp[match[t]] != t  # match[t].shape = [n`]      # jtyp中所有元素均为0，所以肯定等与t，所以这个for循环也等于什么都没做
                # match[t, aaa] = N   # 将索引替换为N，这个索引值是超出最大索引的

            match[cost > 1.5 * 1.5] = N     # 将cost大于阈值的点的配对关系设为N，这个索引值是超出最大索引的（这个1.5的量纲是像素）
            match = match.flatten()

            _ = torch.arange(n_type * K, device=device)
            u, v = torch.meshgrid(_, _)     # 列出了所有可能点的uv坐标，
            u, v = u.flatten(), v.flatten()
            up, vp = match[u], match[v]     # 根据预测点与GT点的配对关系，将端点的uv值up表示一个端点索引， vp表示另一个端点的索引
            # up(vp).shape = K^2, 将网格索引值替换为GT点索引值，也既连接矩阵的索引值，则在up, vp中各取一个对应点，也就对应了连接矩阵中的一个点

            # u或v中索引值为N的点，其值一定是0，所以相当于排除了这些cost过大的点
            label = Lpos[up, vp]     # label.shape = [K^2], [0 or 1, ...]， 根据转换的GT索引值从连接矩阵中取出连接关系

            if mode == "training":
                # torch.nonzero(tensor): 返回tensor中非零元素在每个维度的索引
                c = torch.zeros_like(label, dtype=torch.bool)

                # sample positive lines
                cdx = label.nonzero().flatten()     # cdx.shape = [非零元素的数量]

                if len(cdx) > M.n_dyn_posl:
                    # print("too many positive lines")  # 正直线过多，应该是一个GT点对应了多个预测点，但是上面不是用过了NMS吗？
                    # torch.randperm(n): 将从0~n-1的整数序列随机打乱，并返回
                    perm = torch.randperm(len(cdx), device=device)[: M.n_dyn_posl]
                    cdx = cdx[perm]
                c[cdx] = 1      # 从所有正样本直线中保留一定数量的positive line

                # sample negative lines
                cdx = Lneg[up, vp].nonzero().flatten()          #  SLS采样的直线中的负样本
                if len(cdx) > M.n_dyn_negl:
                    # print("too many negative lines")
                    perm = torch.randperm(len(cdx), device=device)[: M.n_dyn_negl]
                    cdx = cdx[perm]
                c[cdx] = 1      # 从所有负样本直线中保留一定数量的negative lines

                # sample other (unmatched) lines
                # 从0~len(c)之间随机采样出(M.n_dyn_othr,)的整数数组，也就是(600)的数组，
                # 也既添加600个随机直线，这600根直线里的大部分可能都是两个端点都每预测正确的直线，当然也可能包含上的pos line和 neg line
                cdx = torch.randint(len(c), (M.n_dyn_othr,), device=device)
                c[cdx] = 1
            else:
                c = (u < v).flatten()
            # c里面的值非0即1，表示这根直线是否被选中

            # sample lines
            # c表示直线是否被选中， u，v分别表示被选中直线的两个端点的索引， label表示样本的标签
            u, v, label = u[c], v[c], label[c]  # u(v, label).shape = [被选中直线的数量], u,v的值是整数

            xy = xy.reshape(n_type * K, 2)  # xy是预测点的uv坐标
            xyu, xyv = xy[u], xy[v]     # xyu(xyv).shape = [被选中直线的数量, 2]， 被选中直线的端点的uv坐标

            # print(2222222)
            u2v = xyu - xyv     # uv坐标系下直线的长度
            # print(u2v.shape)
            # print(u2v)

            u2v /= torch.sqrt((u2v ** 2).sum(-1, keepdim=True)).clamp(min=1e-6)     # 相对长度归一化

            # print(sum((u[:, None] > K).float()))

            # print(u2v)

            # feat.shape = (num_chosen_line, 8)
            feat = torch.cat(
                [
                    xyu / 128 * M.use_cood,     # uv coordinate of endpoint 1 of the chosen lines, shape of (num_chosen_line, 2)
                    xyv / 128 * M.use_cood,     # uv coordinate of endpoint 2 of the chosen lines, shape of (num_chosen_line, 2)
                    u2v * M.use_slop,           # Inclination degree(tan)  , shape of (num_chosen_line, 2)
                    (u[:, None] > K).float(),   # whether the index of chosen enpoint is greater than K, shape of (num_chosen_line, 1)
                    (v[:, None] > K).float(),   # K is defined as K = min(int(N * 2 + 2), max_K) durning training
                ],
                dim=1,
            )

            line = torch.cat([xyu[:, None], xyv[:, None]], 1)  # 选出的所有直线， line.shape = [选出的直线数量， 2， 2]


            xy = xy.reshape(n_type, K, 2)  # xy.shape = [1, K, 2]
            jcs = [xy[i, score[i] > 0.03] for i in range(n_type)]   # 留下score大于0.03的点, jcs中点的数量小于或等于K


            # line.shape = [选出的直线数量， 2， 2], label.shape = [选出的直线数量], feat, jcs.shape = [socre>0.03的点数，2]
            return line, label.float(), feat, jcs


def non_maximum_suppression(a):
    # 通过最大池化实现极大值抑制
    # 那么每个5*5的方格红只能留下一个交点（多个极大值点的5*5会有交叠）
    ap = F.max_pool2d(a, 3, stride=1, padding=1)    #
    mask = (a == ap).float().clamp(min=0.0)     # 非极大值点设置为0
    # plt.subplot(131)
    # plt.imshow(a.squeeze(0).cpu())
    # plt.subplot(132)
    # plt.imshow(ap.squeeze(0).cpu())
    # plt.subplot(133)
    # plt.imshow(mask.squeeze(0).cpu())
    # plt.show()
    return a * mask     # 只返回交点


class Bottleneck1D(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Bottleneck1D, self).__init__()

        planes = outplanes // 2
        self.op = nn.Sequential(
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv1d(inplanes, planes, kernel_size=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, planes, kernel_size=3, padding=1),
            nn.BatchNorm1d(planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(planes, outplanes, kernel_size=1),
        )

    def forward(self, x):
        return x + self.op(x)
