import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_planes, planes, width, height, kernel_size, time_interval, stride=1, bias=False):
        super().__init__()

        self.in_planes = in_planes
        self.planes = planes
        self.width = width
        self.height = height
        self.kernel_size = kernel_size
        self.time_interval = time_interval
        self.stride = stride
        self.bias = bias

        self.f = nn.Conv2d(self.in_planes, self.planes, self.kernel_size, stride=self.stride, bias=self.bias)
        self.s = torch.zeros((self.time_interval, 1, self.planes, self.height, self.width)).to('cuda:0')

    def forward(self, t, x):
        self.s[t] = self.f(x[t])
        return self.s


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride, planes, width, height, time_interval):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.planes = planes
        self.width = width
        self.height = height
        self.time_interval = time_interval

        self.f = nn.AvgPool2d(self.kernel_size, self.stride)
        self.s = torch.zeros((self.time_interval, 1, self.planes, self.height, self.width)).to('cuda:0')

    def forward(self, t, x):
        self.s[t] = self.f(x[t])
        return self.s


class Linear(nn.Module):
    def __init__(self, in_neurons, neurons, time_interval, bias=False):
        super().__init__()

        self.in_neurons = in_neurons
        self.neurons = neurons
        self.time_interval = time_interval
        self.bias = bias

        self.f = nn.Linear(self.in_neurons, self.neurons, self.bias)
        self.s = torch.zeros((self.time_interval, 1, self.neurons)).to('cuda:0')

    def forward(self, t, x):
        self.s[t] = self.f(x[t])
        return self.s


class IF1d(nn.Module):
    def __init__(self, neurons, time_interval, leak=0.0, threshold=1.0, resting=0.0, v_min=None):
        super().__init__()

        self.neurons = neurons
        self.time_interval = time_interval
        self.v_leak = leak
        self.v_th = threshold
        self.v_reset = resting
        if v_min is None:
            self.v_min = -10.0 * self.v_th
        else:
            self.v_min = self.v_reset

        self.v = torch.zeros(self.neurons).to('cuda:0')
        # Now, batch_size is always 1
        self.s = torch.zeros((self.time_interval, 1, self.neurons)).to('cuda:0')

    def forward(self, t, x):
        # t: scalar
        # x.size: [time_interval, batch_size=1, neurons]
        # s.size: [time_interval, batch_size=1, neurons]
        self.v += self.v_leak + x[t, 0]
        self.s[t, 0, self.v >= self.v_th] = 1
        self.v[self.v >= self.v_th] = self.v_reset
        self.v[self.v < self.v_min] = self.v_min
        return self.s


class IF2d(nn.Module):
    def __init__(self, planes, height, width, time_interval, leak=0.0, threshold=1.0, resting=0.0, v_min=None):
        super().__init__()
        self.planes = planes
        self.height = height
        self.width = width
        self.time_interval = time_interval
        self.v_leak = leak
        self.v_th = threshold
        self.v_reset = resting
        if v_min is None:
            self.v_min = -10.0 * self.v_th
        else:
            self.v_min = self.v_reset

        self.v = torch.zeros((self.planes, self.height, self.width)).to('cuda:0')
        # Now, batch_size is always 1
        self.s = torch.zeros((self.time_interval, 1, self.planes, self.height, self.width)).to('cuda:0')

    def forward(self, t, x):
        # t: scalar
        # x.size: [time_interval, batch_size=1, channels, height, width]
        # s.size: [time_interval, batch_size=1, channels, height, width]
        self.v += self.v_leak + x[t, 0]
        self.s[t, 0, self.v >= self.v_th] = 1
        self.v[self.v >= self.v_th] = self.v_reset
        self.v[self.v < self.v_min] = self.v_min
        return self.s
