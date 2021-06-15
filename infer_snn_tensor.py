import time
import argparse

import numpy as np

import torch
import torchvision

from n3ml.network import TailoredCNN, SpikingCNN
from n3ml.util import CIFAR10_SpikeGenerator, SpikeCounter


def load_model(pretrained, num_classes, time_interval):
    model = TailoredCNN()
    model.load_state_dict(torch.load(pretrained)['model'])

    named_param = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            named_param[name] = param.data.detach().cpu().numpy()

    model = TailoredCNN.make_snn(num_classes=num_classes, time_interval=time_interval)

    model.extractor[0].weight.data.copy_(torch.Tensor(named_param['extractor.0.weight']))
    model.extractor[3].weight.data.copy_(torch.Tensor(named_param['extractor.3.weight']))
    model.extractor[6].weight.data.copy_(torch.Tensor(named_param['extractor.6.weight']))
    model.classifier[0].weight.data.copy_(torch.Tensor(named_param['classifier.0.weight']))
    model.classifier[2].weight.data.copy_(torch.Tensor(named_param['classifier.2.weight']))

    return model


def validate(val_loader, model, generator, counter, time_interval):
    start = time.time()
    for image, label in val_loader:
        image = image.cuda()
        label = label.cuda()

        for t in range(time_interval):
            spiked_image = generator(t, image)

            o = model(t, spiked_image)

        print(o.size())
    end = time.time()
    print("on gpu elapsed times: {}".format(end - start))


def app(opt):
    print(opt)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(24),
                torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size)

    # model = load_model(opt.pretrained, opt.num_classes, opt.time_interval)
    model = SpikingCNN()
    model.cuda()

    generator = CIFAR10_SpikeGenerator(3, 24, 24, opt.time_interval)
    generator.cuda()

    counter = SpikeCounter()

    start = time.time()
    val_acc = validate(val_loader, model, generator, counter, opt.time_interval)
    end = time.time()

    print("elapsed: {} - val_acc: {}".format(end-start, val_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--time_interval', default=300, type=int)
    parser.add_argument('--pretrained', default='pretrained/tailored_cnn.pt')

    app(parser.parse_args())
