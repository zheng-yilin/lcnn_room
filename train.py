# #!/usr/bin/env python3
"""Train L-CNN
Usage:
    train.py [options] <yaml-config>
    train.py (-h | --help )

Arguments:
   <yaml-config>                   Path to the yaml hyper-parameter file

Options:
   -h --help                       Show this screen.
   -d --devices <devices>          Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>    Folder identifier [default: default-identifier]
"""

import datetime
import glob
import os
import os.path as osp
import platform
import pprint
import random
import shlex
import shutil
import signal
import subprocess
import sys
import threading

import numpy as np
import torch
import yaml
from docopt import docopt

import lcnn
from lcnn.config import C, M
from lcnn.datasets import WireframeDataset, collate
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def get_outdir(identifier):
    # load config
    name = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    # name += "-%s" % git_hash()
    name += "-%s" % identifier
    outdir = osp.join(osp.expanduser(C.io.logdir), name)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    C.io.resume_from = outdir
    C.to_yaml(osp.join(outdir, "config.yaml"))
    os.system(f"git diff HEAD > {outdir}/gitdiff.patch")
    return outdir       # outdir: logs/211117-102511-baseline


def main():
    # args = docopt(__doc__)        # 从.py开头的注释中获取参数dict
    # config_file = args["<yaml-config>"] or "config/wireframe.yaml"

    # config_file = "config/wireframe.yaml"
    config_file = "config/wireframe_room.yaml"
    C.update(C.from_yaml(filename=config_file))     # 所有参数
    M.update(C.model)                   # 与模型有关的参数
    # pprint.pprint(C, indent=4)      # 打印参数
    resume_from = C.io.resume_from      # None

    # WARNING: L-CNN is still not deterministic
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True       # 如果网络的结构和输入数据的维度是确定的，那么可以在训练前先选择出最适合的计算方法，以大幅减少总体训练时间
        # torch.backends.cudnn.benchmark = True         # 这两个有些相似，都是确定卷积算法以提升训练速度
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    # 1. dataset
    # uncomment for debug DataLoader
    # wireframe.datasets.WireframeDataset(datadir, split="train")[0]
    # sys.exit(0)

    datadir = C.io.datadir
    kwargs = {
        "collate_fn": collate,
        "num_workers": C.io.num_workers if os.name != "nt" else 0,  # os.name可以用来判断正在使用的平台，windows返回nt， linux返回posix
        "pin_memory": True,
    }

    train_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="training"),
        # WireframeDataset(datadir, split="train"),
        shuffle=True,
        batch_size=M.batch_size,
        **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        WireframeDataset(datadir, split="validation"),
        # WireframeDataset(datadir, split="valid"),
        shuffle=False,
        batch_size=M.batch_size_eval,
        **kwargs,
    )
    epoch_size = len(train_loader)
    # print("epoch_size (train):", epoch_size)        # 20000， 训练集图片数量为20000，但这是经过了数据扩充以后的，一张图片变4张（当然验证集的就不用扩充）
    # print("epoch_size (valid):", len(val_loader))   # 462，测试集不用进行数据扩充

    if resume_from:     # 发生意外，恢复训练, resume_from：是否要进行训练恢复
        checkpoint = torch.load(osp.join(resume_from, "checkpoint_latest.pth"))

    # 2. model
    if M.backbone == "stacked_hourglass":
        model = lcnn.models.hg(
            depth=M.depth,
            head=MultitaskHead,
            num_stacks=M.num_stacks,
            num_blocks=M.num_blocks,
            num_classes=sum(sum(M.head_size, [])),      # head_size: [[2], [1], [2]]
        )

    else:
        raise NotImplementedError

    model = MultitaskLearner(model)     # junction proposal module
    model = LineVectorizer(model)       # line sample module

    if resume_from:
        model.load_state_dict(checkpoint["model_state_dict"])
    # model = torch.nn.DataParallel(model)
    model = model.to(device)

    # 3. optimizer
    if C.optim.name == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=C.optim.lr,
            weight_decay=C.optim.weight_decay,
            amsgrad=C.optim.amsgrad,
        )
    elif C.optim.name == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=C.optim.lr,
            weight_decay=C.optim.weight_decay,
            momentum=C.optim.momentum,
        )
    else:
        raise NotImplementedError

    if resume_from:
        optim.load_state_dict(checkpoint["optim_state_dict"])

    # outdir = resume_from or get_outdir(args["--identifier"])
    outdir = resume_from or get_outdir('baseline')
    # print("outdir:", outdir)     # outdir: logs/211117-102511-baselin

    try:
        trainer = lcnn.trainer.Trainer(
            device=device,
            model=model,
            optimizer=optim,
            train_loader=train_loader,
            val_loader=val_loader,
            out=outdir,
        )
        if resume_from:
            trainer.iteration = checkpoint["iteration"]
            if trainer.iteration % epoch_size != 0:
                print("WARNING: iteration is not a multiple of epoch_size, reset it")
                trainer.iteration -= trainer.iteration % epoch_size
            trainer.best_mean_loss = checkpoint["best_mean_loss"]
            del checkpoint

        trainer.train()

    except BaseException:
        if len(glob.glob(f"{outdir}/viz/*")) <= 1:
            shutil.rmtree(outdir)   # 递归删除多级目录
        raise


if __name__ == "__main__":
    main()
