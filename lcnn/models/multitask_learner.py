from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lcnn.config import M


class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class):  # input_channels = 256, num_class = 5
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)     # m = 256/4 = 64
        heads = []
        for output_channels in sum(M.head_size, []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)   # len(self.heads) = 3
        assert num_class == sum(sum(M.head_size, []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class MultitaskLearner(nn.Module):
    def __init__(self, backbone):
        super(MultitaskLearner, self).__init__()
        self.backbone = backbone
        head_size = M.head_size     # [[2], [1], [2]]
        self.num_class = sum(sum(head_size, []))    # self.num_class = 5
        self.head_off = np.cumsum([sum(h) for h in head_size])      # self.head_off = [2, 3, 5]

    def forward(self, input_dict):
        image = input_dict["image"]

        # outputs是两个预测头（J，O）的输出，第一个元素是第二个hg网络的输出，第二个元素是第一个沙漏网络的输出,两个元素的shape都是(ba, 5, 128, 128)
        # feature是第二个沙漏网络最后输出的特征图, (bs, 256, 128, 128)
        outputs, feature = self.backbone(image)     # out[::-1], y
        result = {"feature": feature}

        # print(feature[1:].shape)
        # m_value, m_index = torch.max(feature.squeeze(0), dim=0)
        # print(m_value.shape)
        #
        # plt.imshow(m_value.cpu().numpy())
        # plt.show()

        batch, channel, row, col = outputs[0].shape     # (1, 5, 128, 128)

        T = input_dict["target"].copy()    # target中包含["jmap", "joff", "lmap"]
        # print('shape of target:')
        # print(T['jmap'].shape)      # bs=2, torch.Size([2, 1, 128, 128])     , bs=1, torch.Size([1, 1, 128, 128])
        # print(T['joff'].shape)      #       torch.Size([2, 1, 2, 128, 128])          torch.Size([1, 1, 2, 128, 128])
        # print(T['lmap'].shape)      #       torch.Size([2, 128, 128])                torch.Size([1, 128, 128])

        n_jtyp = T["jmap"].shape[1]     # n_jtyp = 1

        # switch to CNHW
        for task in ["jmap"]:
            T[task] = T[task].permute(1, 0, 2, 3)   # (b, c, h, w)-->(c, b, h, w)
        for task in ["joff"]:
            T[task] = T[task].permute(1, 2, 0, 3, 4)    # (2, 1, 2, 128, 128)-->(1, 2, 2, 128, 128)
                                                        # 这里的joff表示两个方向上的偏移量，所以原本的维度为(b, c, (2, h, w))

        offset = self.head_off  # [2, 3, 5]   # 每种监督信息的channel顺序
        loss_weight = M.loss_weight     # {'jmap': 8.0, 'lmap': 0.5, 'joff': 0.25, 'lpos': 1, 'lneg': 1}

        losses = []     # 这个for循环，实现对两个hg module的输出计算损失
        for stack, output in enumerate(outputs):    # stack指第几个沙漏网络，output表示对应沙漏网络的输出
            # tensor.contiguous()：对于经过维度变化后内存上不连续（但语义上连续）的张量，重新开辟一块内存，使得该张量内存上也连续
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()     # (5, bs, h, w),len(output) = 5, 这个reshape()多此一举啊

            jmap = output[0 : offset[0]].reshape(n_jtyp, 2, batch, row, col)    # (1, 2, bs, h, w), output[0:2]，reshape操作不就是等于unsqueeze(0)么
            lmap = output[offset[0] : offset[1]].squeeze(0)     # (bs, 128, 128), output[2:3]， 这个squeeze是把channel这个维度去掉了
            joff = output[offset[1] : offset[2]].reshape(n_jtyp, 2, batch, row, col)    # (1, 2, bs, 128, 128), output[3:5]
            print('in multitask learner')
            print(jmap)
            print(joff)

            print('GT')
            # print(torch.max(T["jmap"][0]))
            # print(torch.max(T["joff"][0]))
            # plt.subplot(121)
            # plt.imshow(T["jmap"][0][0].cpu().numpy())
            # plt.subplot(122)
            # plt.imshow(T["joff"][0][0][0].cpu().numpy())
            # plt.show()


            if stack == 0:
                # result["preds"]是最后一个网络的输出,
                result["preds"] = {
                    # (b, c, h, w)-(c, b, h, w)-(j, c, b, h, w)-(b, j, h, w)
                    # 这里通过最后的索引，在c这个维度上只取了第二个通道，那为什么网络要预测出两个通道呢？因为作者在这个工作中没有对交点类型进行分别
                    "jmap": jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],

                    # (b, c, h, w)-(c, b, h, w)-(b, h, w), c=1所以可以直接squeeze掉
                    "lmap": lmap.sigmoid(),

                    # (b, c, h, w)-(c, b, h, w)-(j, c, b, h, w)-(b, j, c, h, w)
                    "joff": joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5,
                }
                if input_dict["mode"] == "testing":     # 测试模式下只计算第二个hg网络的预测结果
                    return result





            L = OrderedDict()
            # jmap计算交叉熵损失函数，但是就以一个像素来计算，感觉比较难收敛，感觉可以对GT jmap进行高斯模糊
            L["jmap"] = sum(
                cross_entropy_loss(jmap[i], T["jmap"][i]) for i in range(n_jtyp)    # n_jtyp = 1，这个 for 循环等于没加
                # cross_entropy_loss(jmap[0], T["jmap"][0])    # jmap计算交叉熵损失函数
            )

            # lmap计算二分类交叉熵损失函数
            L["lmap"] = (
                F.binary_cross_entropy_with_logits(lmap, T["lmap"], reduction="none").mean(2).mean(1)
            )
            # print(12121)
            # print(T["jmap"][0].shape)
            # plt.imshow(T["jmap"][0][0].cpu())
            # plt.show()

            # 偏移量计算L1 loss
            L["joff"] = sum(
                sigmoid_l1_loss(joff[i, j], T["joff"][i, j], -0.5, T["jmap"][i])
                for i in range(n_jtyp)
                for j in range(2)
            )
            for loss_name in L:
                L[loss_name].mul_(loss_weight[loss_name])   # 损失加权
            losses.append(L)
            # print(losses)
        result["losses"] = losses   # 这个losses包含了两个hg module的输出
        '''losses
        [OrderedDict([('jmap', tensor([0.6529, 0.6508], device='cuda:0', grad_fn=<MulBackward0>)), 
                      ('lmap', tensor([0.1423, 0.1652], device='cuda:0', grad_fn=<MulBackward0>)), 
                      ('joff', tensor([0.1230, 0.1317], device='cuda:0', grad_fn=<MulBackward0>))]),
         OrderedDict([('jmap', tensor([0.8486, 0.8866], device='cuda:0', grad_fn=<MulBackward0>)), 
                      ('lmap', tensor([0.1496, 0.1739], device='cuda:0', grad_fn=<MulBackward0>)), 
                      ('joff', tensor([0.1259, 0.1295], device='cuda:0', grad_fn=<MulBackward0>))])]
        '''
        return result


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    nlogp = -F.log_softmax(logits, dim=0)
    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):     # mask.shape = (b, h, w)
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)     # loss.shape = (b, h, w)

    if mask is not None:
        w = mask.mean(2, True).mean(1, True)    # 等同于torch.mean(mask)，w表示的是交点数量与所有像素(128*128)的比例
        w[w == 0] = 1   # 防止除数等于0
        loss = loss * (mask / w)    # 增大真正有交点处的权重

    return loss.mean(2).mean(1)
