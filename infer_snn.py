import time
import argparse

import numpy as np

import torch
import torch.nn as nn
import torchvision


class Conv2d:
    def __init__(self, in_planes, planes, kernerl_size, stride=1, bias=False):
        self.in_planes = in_planes
        self.planes = planes
        self.kernerl_size = kernerl_size
        self.stride = stride
        self.bias = bias

        self.conv2d = nn.Conv2d(self.in_planes, self.planes, self.kernerl_size, stride=self.stride, bias=self.bias)

    def __call__(self, x):
        x = torch.Tensor(x).unsqueeze(0)
        o = self.conv2d(x).squeeze(0)
        return o.detach().cpu().numpy()


class AvgPool2d:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

        self.avgpool2d = nn.AvgPool2d(self.kernel_size, self.stride)

    def __call__(self, x):
        x = torch.Tensor(x).unsqueeze(0)
        o = self.avgpool2d(x).squeeze(0)
        return o.detach().cpu().numpy()


class Linear:
    def __init__(self, in_neurons, neurons, bias=False):
        self.in_neurons = in_neurons
        self.neurons = neurons
        self.bias = bias

        self.linear = nn.Linear(self.in_neurons, self.neurons, self.bias)

    def __call__(self, x):
        x = torch.Tensor(x).unsqueeze(0)
        o = self.linear(x).squeeze(0)
        return o.detach().cpu().numpy()


class IF:
    def __init__(self):
        raise NotImplementedError

    def __call__(self, x):
        f = np.zeros(self.v.shape)
        self.v[:] = self.v + self.leak + x
        f[self.v >= self.th] = 1
        self.v[self.v >= self.th] = self.reset
        self.v[self.v < self.v_min] = self.v_min
        return f


class IF1d(IF):
    def __init__(self, neurons, leak=0.0, threshold=1.0, resting=0.0, v_min=None):
        self.neruons = neurons
        self.leak = leak
        self.th = threshold
        self.reset = resting
        if v_min is None:
            self.v_min = -10.0 * self.th

        self.v = np.zeros(self.neruons)


class IF2d(IF):
    def __init__(self, planes, width, height, leak=0.0, threshold=1.0, resting=0.0, v_min=None):
        self.planes = planes
        self.width = width
        self.height = height
        self.leak = leak
        self.th = threshold
        self.reset = resting
        if v_min is None:
            self.v_min = -10.0 * self.th

        self.v = np.zeros((self.planes, self.width, self.height))



class TailoredCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        in_channels = 3
        out_channels = 64
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(out_channels, out_channels, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(out_channels, out_channels, 3, bias=False),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.ReLU(),
            nn.Linear(out_channels, num_classes, bias=False)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class SpikingCNN:
    def __init__(self):
        self.conv1 = Conv2d(3, 64, 5)
        self.if1 = IF2d(64, 20, 20, threshold=5)
        self.sub1 = AvgPool2d(2, 2)
        self.conv2 = Conv2d(64, 64, 5)
        self.if2 = IF2d(64, 6, 6, threshold=0.99)
        self.sub2 = AvgPool2d(2, 2)
        self.conv3 = Conv2d(64, 64, 3)
        self.if3 = IF2d(64, 1, 1)
        self.fc1 = Linear(64, 64)
        self.if4 = IF1d(64, threshold=0.99)
        self.fc2 = Linear(64, 10)
        self.if5 = IF1d(10)

    def __call__(self, x):
        o = self.if1(self.conv1(x))
        o = self.sub1(o)
        o = self.if2(self.conv2(o))
        o = self.sub2(o)
        o = self.if3(self.conv3(o))
        o = o.flatten()
        o = self.if4(self.fc1(o))
        o = self.if5(self.fc2(o))
        return o


def load_model(pretrained):
    model = TailoredCNN()
    model.load_state_dict(torch.load(pretrained)['model'])

    named_param = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            named_param[name] = param.data.detach().cpu().numpy()

    # w = model.conv1.conv2d.weight.data.detach().cpu().numpy()
    # model.conv1.conv2d.weight.data.copy_()

    model = SpikingCNN()
    model.conv1.conv2d.weight.data.copy_(torch.Tensor(named_param['extractor.0.weight']))
    model.conv2.conv2d.weight.data.copy_(torch.Tensor(named_param['extractor.3.weight']))
    model.conv3.conv2d.weight.data.copy_(torch.Tensor(named_param['extractor.6.weight']))
    model.fc1.linear.weight.data.copy_(torch.Tensor(named_param['classifier.0.weight']))
    model.fc2.linear.weight.data.copy_(torch.Tensor(named_param['classifier.2.weight']))
    # model.conv1.conv2d.weight.data.copy_(torch.Tensor(named_param['conv1.weight']))
    # model.conv2.conv2d.weight.data.copy_(torch.Tensor(named_param['conv2.weight']))
    # model.conv3.conv2d.weight.data.copy_(torch.Tensor(named_param['conv3.weight']))
    # model.fc1.linear.weight.data.copy_(torch.Tensor(named_param['fc1.weight']))
    # model.fc2.linear.weight.data.copy_(torch.Tensor(named_param['fc2.weight']))

    return model


class SpikeGenerator:
    def __init__(self, c=0.3):
        self.c = c

    def __call__(self, x):
        r = np.random.uniform(0, 1, x.shape)
        f = np.zeros(x.shape)
        f[self.c*x > r] = 1
        return f


class CIFAR10_SpikeGenerator:
    def __init__(self, channel, height, width, threshold=1):
        self.channel = channel
        self.height = height
        self.width = width
        self.th = threshold
        self.v_min = -10*self.th

        self.v = np.zeros((channel, height, width))

    def __call__(self, x):
        assert self.v.shape == x.shape
        self.v[:] = self.v + x
        f = np.zeros(self.v.shape)
        f[self.v >= self.th] = 1
        self.v[self.v >= self.th] = self.v[self.v >= self.th] - self.th
        self.v[self.v < self.v_min] = self.v_min
        return f


class SpikeCounter:
    def __init__(self):
        pass

    def __call__(self, x):
        o = np.zeros(x[0].shape[0])
        for v in x:
            o += v
        return o


def validate(val_loader, model, generator, counter, time_interval):
    total_images = 0
    num_corrects = 0
    i = 0

    start = time.time()

    for image, label in val_loader:
        image = image.squeeze(0).detach().cpu().numpy()
        label = label.squeeze(0).detach().cpu().numpy()

        out = []

        for _ in range(time_interval):
            spiked_image = generator(image)

            out.append(model(spiked_image))

        print("out.size: {}".format(len(out)))

    end = time.time()
    print("on cpu elapsed times: {}".format(end - start))

    #     pred = counter(out)
    #
    #     total_images += 1
    #     if label == np.argmax(pred):
    #         num_corrects += 1
    #
    #     if i % 100 == 0:
    #         print("step: {} - acc: {}".format(i, num_corrects/total_images))
    #     i += 1
    #
    # val_acc = num_corrects / total_images
    #
    # return val_acc


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

    model = load_model(opt.pretrained)

    criterion = None

    generator = CIFAR10_SpikeGenerator(3, 24, 24)

    counter = SpikeCounter()

    start = time.time()
    val_acc = validate(val_loader, model, generator, counter, opt.time_interval)
    end = time.time()

    print("elapsed: {} - val_acc: {}".format(end-start, val_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--time_interval', default=300, type=int)
    parser.add_argument('--pretrained', default='pretrained/tailored_cnn.pt')

    app(parser.parse_args())