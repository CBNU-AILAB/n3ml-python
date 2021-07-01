import time
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from n3ml.model import Hunsberger2015


def app(opt):
    """
        1. pretrained model을 불러온다.
        2. 해당 모델 구조에서 soft LIF를 LIF로 대체한다.
        3. 추론한다.
    """

    model = Hunsberger2015(num_classes=opt.num_classes, amplitude=opt.amplitude, tau_ref=opt.tau_ref,
                           tau_rc=opt.tau_rc, gain=opt.gain, sigma=opt.sigma).cuda()

    for l in model.children():
        print(l)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', default=10, type=int)

    parser.add_argument('--amplitude', default=0.063, type=float)
    parser.add_argument('--tau_ref', default=0.001, type=float)
    parser.add_argument('--tau_rc', default=0.05, type=float)
    parser.add_argument('--gain', default=0.825, type=float)
    parser.add_argument('--sigma', default=0.02, type=float)

    app(parser.parse_args())
