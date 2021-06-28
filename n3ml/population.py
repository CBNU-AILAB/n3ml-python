from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable, Function

import matplotlib.pyplot as plt


class BasePopulation(nn.Module):
    def __init__(self):
        super().__init__()

    def get(self, name):
        # self.__dict__을 출력해보면 '_parameters'와 '_buffers'를 이름으로 가지는 keys가 있다.
        # 현재 뉴런 모델에서 사용되는 변수를 모두 _buffers에 저장하였지만 approximate BP 알고리즘을 사용하는 경우
        # potential이나 일부 변수가 _parameters에 저장되어야 한다. 그렇게 해야 .backward()에서 사용이 가능하다.
        return self.__dict__['_buffers'][name].numpy()


class InputPopulation(BasePopulation):
    def __init__(self,
                 neurons,
                 dt=1.,
                 traces=True,
                 tau_tr=20.,
                 scale_tr=1.):
        super().__init__()

        self.neurons = neurons
        self.traces = traces

        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('s', torch.zeros(neurons))
        if traces:
            self.register_buffer('x', torch.zeros(neurons))
            self.register_buffer('tau_tr', torch.tensor(tau_tr))
            self.register_buffer('scale_tr', torch.tensor(scale_tr))

    def init_param(self):
        if self.traces:
            self.x.fill_(0.)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        self.s[:] = x

        if self.traces:
            self.x[:] *= torch.exp(-self.dt / self.tau_tr)
            self.x[:] += self.scale_tr * self.s

        return self.s


class LIFPopulation(BasePopulation):
    def __init__(self,
                 neurons: int,
                 dt: float = 1.0,
                 tau_rc: float = 100.0,
                 v_th: float = -52.0,
                 rest: float = -65.0,
                 reset: float = -65.0,
                 tau_ref: float = 5.0,
                 traces: bool = True,
                 tau_tr: float = 20.0,
                 scale_tr: float = 1.0) -> None:
        super().__init__()

        self.neurons = neurons
        self.traces = traces

        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('v', torch.zeros(neurons))
        self.register_buffer('tau_rc', torch.tensor(tau_rc))
        self.register_buffer('v_th', torch.tensor(v_th))
        self.register_buffer('s', torch.zeros(neurons))
        self.register_buffer('rest', torch.tensor(rest))
        self.register_buffer('reset', torch.tensor(reset))
        self.register_buffer('refrac', torch.zeros(neurons))
        self.register_buffer('tau_ref', torch.tensor(tau_ref))
        if traces:
            self.register_buffer('x', torch.zeros(neurons))
            self.register_buffer('tau_tr', torch.tensor(tau_tr))
            self.register_buffer('scale_tr', torch.tensor(scale_tr))

    def init_param(self):
        self.v.fill_(self.reset)
        if self.traces:
            self.x.fill_(0.)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        self.v[:] = torch.exp(-self.dt / self.tau_rc) * (self.v - self.rest) + self.rest

        self.v += (self.refrac <= 0).float() * x

        self.refrac -= self.dt

        self.s[:] = self.v >= self.v_th

        self.refrac.masked_fill_(self.s.bool(), self.tau_ref)
        self.v.masked_fill_(self.s.bool(), self.rest)

        # Update spike traces
        if self.traces:
            self.x[:] *= torch.exp(-self.dt / self.tau_tr)
            self.x[:] += torch.scale_tr * self.s

        return self.s


class DiehlAndCookPopulation(BasePopulation):
    def __init__(self,
                 neurons: int,
                 dt: float = 1.0,
                 tau_rc: float = 100.0,
                 v_th: float = -52.0,
                 theta: float = 0.05,
                 rest: float = -65.0,
                 reset: float = -65.0,
                 tau_ref: float = 5.0,
                 traces: bool = True,
                 tau_tr: float = 20.0,
                 scale_tr: float = 1.0):
        super().__init__()

        self.neurons = neurons
        self.traces = traces
        self.reset_theta = theta

        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('v', torch.zeros(neurons))
        self.register_buffer('tau_rc', torch.tensor(tau_rc))
        self.register_buffer('v_th', torch.tensor(v_th))
        self.register_buffer('theta', torch.zeros(neurons))
        self.register_buffer('s', torch.zeros(neurons))
        self.register_buffer('rest', torch.tensor(rest))
        self.register_buffer('reset', torch.tensor(reset))
        self.register_buffer('refrac', torch.zeros(neurons))
        self.register_buffer('tau_ref', torch.tensor(tau_ref))
        if traces:
            # TODO: 초기화는 어떤 값으로 해야 하는가?
            self.register_buffer('x', torch.zeros(neurons))
            self.register_buffer('tau_tr', torch.tensor(tau_tr))
            self.register_buffer('scale_tr', torch.tensor(scale_tr))

    def init_param(self):
        self.v.fill_(self.reset)
        self.theta.fill_(self.reset_theta)
        if self.traces:
            self.x.fill_(0.)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        self.v[:] = torch.exp(-self.dt / self.tau_rc) * (self.v - self.rest) + self.rest

        self.v += (self.refrac <= 0).float() * x

        self.refrac -= self.dt

        self.s[:] = self.v >= self.v_th + self.theta

        self.refrac.masked_fill_(self.s.bool(), self.tau_ref)
        self.v.masked_fill_(self.s.bool(), self.rest)

        # Update adaptive threshold
        self.theta += self.s

        # Update spike traces
        if self.traces:
            self.x[:] *= torch.exp(-self.dt / self.tau_tr)
            self.x[:] += self.scale_tr * self.s

        return self.s


def softplus(x, sigma=1.):
    y = torch.true_divide(x, sigma)
    z = x.clone().float()
    z[y < 34.0] = sigma * torch.log1p(torch.exp(y[y < 34.0]))
    return z


def lif_j(j, tau_ref, tau_rc, amplitude=1.):
    j = torch.true_divide(1., j)
    j = torch.log1p(j)
    j = tau_ref + tau_rc * j
    j = torch.true_divide(amplitude, j)
    return j


class _SoftLIF(Function):
    @staticmethod
    def forward(ctx, x, gain, bias, sigma, v_th, tau_ref, tau_rc, amplitude):
        ctx.save_for_backward(x, gain, bias, sigma, v_th, tau_ref, tau_rc, amplitude)
        # j = gain * x + bias - v_th
        j = gain * x
        j = softplus(j, sigma)
        o = torch.zeros_like(j)
        o[j > 0] = lif_j(j[j > 0], tau_ref, tau_rc, amplitude)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        x, gain, bias, sigma, v_th, tau_ref, tau_rc, amplitude = ctx.saved_tensors
        # y = gain * x + bias - v_th  # TODO: 1이 v_th=1로 했기 때문에 1인 건가? 아니면 다른 것에 의한 건가?
        y = gain * x
        j = softplus(y, sigma)
        yy = y[j > 1e-15]
        jj = j[j > 1e-15]
        vv = lif_j(jj, tau_ref, tau_rc, amplitude)
        d = torch.zeros_like(j)
        d[j > 1e-15] = torch.true_divide((gain * tau_rc * vv * vv),
                                         (amplitude * jj * (jj + 1) * (1 + torch.exp(torch.true_divide(-yy, sigma)))))
        grad_input = grad_output * d
        return grad_input, None, None, None, None, None, None, None


class SoftLIF(BasePopulation):
    def __init__(self, gain=1., bias=0., sigma=0.02, v_th=1., tau_ref=0.001, tau_rc=0.05, amplitude=1.):
        super().__init__()
        self.gain = Variable(torch.tensor(gain), requires_grad=False)
        self.bias = Variable(torch.tensor(bias), requires_grad=False)
        self.sigma = Variable(torch.tensor(sigma), requires_grad=False)
        self.v_th = Variable(torch.tensor(v_th), requires_grad=False)
        self.tau_ref = Variable(torch.tensor(tau_ref), requires_grad=False)
        self.tau_rc = Variable(torch.tensor(tau_rc), requires_grad=False)
        self.amplitude = Variable(torch.tensor(amplitude), requires_grad=False)

    def forward(self, x):
        return _SoftLIF.apply(x, self.gain, self.bias, self.sigma, self.v_th, self.tau_ref, self.tau_rc, self.amplitude)


if __name__ == '__main__':
    lif = LIFPopulation(1)

    num_steps = 1000

    x = torch.randint(0, 2, (num_steps,))
    print(x)

    fig, ax = plt.subplots(2)

    o_s = []
    o_v = []
    for t in range(num_steps):
        s, v = lif(x[t])
        o_v.append(v.clone())
        o_s.append(s.clone())

    ax[0].plot(o_v)
    ax[1].plot(o_s)
    plt.show()
