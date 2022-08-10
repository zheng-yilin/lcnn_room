"""
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) Yichao Zhou (LCNN)
(c) YANG, Wei
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HourglassNet", "hg"]


class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):   # block = Bottleneck2D, num_blocks = 1, planes = 128, depth = 4
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):      # num_blocks = 1
            layers.append(block(planes * block.expansion, planes))      # 就一个Bottleneck2D层
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):  # depth = 4
            res = []    # 第一个res[]里包含了4个残差层（Bottleneck2D）， 后三个res[]里只包含了3个残差层
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        # print(11111111111111)
        # print(nn.ModuleList(hg)[2][0])
        # print(nn.ModuleList(hg)[2][1])

        return nn.ModuleList(hg)    # 所以hg里面一共有13个Bottleneck2D，每个Bottleneck都是一样的，形式为[[4], [3], [3], [3]]

    def _hour_glass_forward(self, n, x):    # n=4, x是输入图片（或多维特征图）
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)  # 每次都要调用4次_hour_glass_forward，函内还需要递归调用三次


class HourglassNet(nn.Module):
    """Hourglass model from Newell et al ECCV 2016"""
    '''
    model = HourglassNet(
        Bottleneck2D,                                                               # block = Bottleneck2D
        head=kwargs.get("head", lambda c_in, c_out: nn.Conv2D(c_in, c_out, 1)),     # <class 'lcnn.models.multitask_learner.MultitaskHead'>
        depth=kwargs["depth"],                                                      # 4
        num_stacks=kwargs["num_stacks"],                                            # 2
        num_blocks=kwargs["num_blocks"],                                            # 1
        num_classes=kwargs["num_classes"],                                          # 5 ,   head_size: [[2], [1], [2]]
        )
    '''

    # block = Bottleneck2D, head = <class 'lcnn.models.multitask_learner.MultitaskHead'>,
    #                                 4,        2,          1,          5
    def __init__(self, block, head, depth, num_stacks, num_blocks, num_classes):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3)    # 1/2
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)      # 这三个layer都没使得特征图尺寸变小
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)    # 1/4

        # build hourglass modules
        ch = self.num_feats * block.expansion   # block.expansion = 2, ch = 256
        # vpts = []
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):     # num_stack = 2
            hg.append(Hourglass(block, num_blocks, self.num_feats, depth))      # block = Bottleneck2D, num_block = 1, num_feats = 128, depth = 4
            res.append(self._make_residual(block, self.num_feats, num_blocks))  # num_block即为步长，所以特征图大小也没有变化
            fc.append(self._make_fc(ch, ch))    # 一个1x1卷积
            score.append(head(ch, num_classes))     # ch = 256, num_classes = 5, 这个score就是用来预测J和O的
            # vpts.append(VptsHead(ch))
            # vpts.append(nn.Linear(ch, 9))
            # score.append(nn.Conv2d(ch, num_classes, kernel_size=1))
            # score[i].bias.data[0] += 4.6
            # score[i].bias.data[2] += 4.6
            if i < num_stacks - 1:  # num_stacks = 2, 所以if字句只会在第一次循环时被执行
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1))   # num_classes = 5
                # fc_ = [conv2d(256, 256, (1, 1))]
                #score_ = [con2d(5, 256, (1, 1))]
        self.hg = nn.ModuleList(hg)     # ML[ML[ml[4], ml[3], ml[3], ml[3]], ML[ml[4], ml[3], ml[3], ml[3]]]

        self.res = nn.ModuleList(res)   # 两个残差块
        self.fc = nn.ModuleList(fc)     # 不是全连接层，两个卷积核为1的二维卷积
        self.score = nn.ModuleList(score)   # 两个卷积核为1的二维卷积层
        # self.vpts = nn.ModuleList(vpts)
        self.fc_ = nn.ModuleList(fc_)       # 一个conv2d(256, 256, (1, 1))
        self.score_ = nn.ModuleList(score_) # 一个con2d(5, 256, (1, 1))

    def _make_residual(self, block, planes, blocks, stride=1):      # block = BottleBlock2D, planes = 64, blocks = 1,
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        out = []
        # out_vps = []
        x = self.conv1(x)   # 1/2
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)     # 1/4
        x = self.layer2(x)
        x= self.layer3(x)

        # 图片先经过几个残差layer，之后再进入的Hourglass
        for i in range(self.num_stacks):    # num_stacks = 2
            y = self.hg[i](x)       # 用第一个沙漏网络处理数据
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            # pre_vpts = F.adaptive_avg_pool2d(x, (1, 1))
            # pre_vpts = pre_vpts.reshape(-1, 256)
            # vpts = self.vpts[i](x)
            out.append(score)
            # out_vps.append(vpts)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        # len(out) = 2, out[0].shape = out[1].shape = (1, 5, 128, 128)
        # out的第一个元素是第二个hg网络的输出，第二个元素是第一个沙漏网络的输出
        # y是第二个沙漏网络的第二个模块fc的输出

        # -1将out中的两个元素位置换了一下，现在是第二个module的输出在前，第一个module的输出在后
        return out[::-1], y  # , out_vps[::-1]


def hg(**kwargs):
    '''
     model = lcnn.models.hg(
            depth=M.depth,      # 4
            head=MultitaskHead,
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),
        )
    '''

    model = HourglassNet(
        Bottleneck2D,
        head=kwargs.get("head", lambda c_in, c_out: nn.Conv2D(c_in, c_out, 1)),     # <class 'lcnn.models.multitask_learner.MultitaskHead'>
        depth=kwargs["depth"],              # 4
        num_stacks=kwargs["num_stacks"],    # 2
        num_blocks=kwargs["num_blocks"],    # 1
        num_classes=kwargs["num_classes"],  # 5 ,   head_size: [[2], [1], [2]]
    )

    return model
