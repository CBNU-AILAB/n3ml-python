import torch
import torch.nn as nn


class Layer(nn.Module):
    def __init__(self):
        super().__init__()


class Bohte(Layer):
    def __init__(self,
                 in_neurons: int,
                 out_neurons: int,
                 delays: int = 16,
                 threshold: float = 1.0,
                 time_constant: float = 5.0) -> None:
        super().__init__()

        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.delays = delays

        self.register_buffer('d', torch.zeros(delays))
        self.register_buffer('v', torch.zeros(out_neurons))
        self.register_buffer('v_th', torch.tensor(threshold))
        self.register_buffer('tau_rc', torch.tensor(time_constant))
        self.register_buffer('w', torch.zeros((out_neurons, in_neurons, delays)))
        self.register_buffer('s', torch.zeros(out_neurons))

    def initialize(self, delay=False) -> None:
        if delay:
            self.d[:] = (torch.rand(self.delays) * 10).int()
        # voltage는 초기화할 필요가 없다.
        self.w[:] = torch.rand((self.out_neurons, self.in_neurons, self.delays)) * 0.02
        self.s.fill_(-1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Step 1. Compute spike response y
        y = self.response(t, x, self.d)

        # Step 2. Compute voltage v
        yy = y.unsqueeze(0).repeat(self.out_neurons, 1, 1)
        self.v[:] = (self.w * yy).sum(dim=(1, 2))

        # Step 3. Compute spike time t
        # Note this is a single spike case
        self.s[torch.logical_and(self.s < 0, self.v >= self.v_th)] = t

        return self.s

    def response(self, t: torch.Tensor, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # t: 0-dimensional tensor
        # x: 1-dimensional tensor
        # d: 1-dimensional tensor
        xx = x.unsqueeze(1).repeat(1, self.delays)
        dd = d.unsqueeze(0).repeat(self.in_neurons, 1)
        tt = t - xx - dd
        o = torch.zeros((self.in_neurons, self.delays))
        o[torch.logical_and(xx != -1, tt >= 0)] = (tt * torch.exp(1 - tt / self.tau_rc) / self.tau_rc)[torch.logical_and(xx != -1, tt >= 0)]
        return o


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
