import torch
import torch.nn as nn

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

        self.reset()

    def reset(self):
        self.x.fill_(0.)

    def forward(self, x):
        self.s[:] = x

        if self.traces:
            self.x[:] *= torch.exp(-self.dt / self.tau_tr)
            self.x[:] += self.scale_tr * self.s

        return self.s


class LIFPopulation(BasePopulation):
    def __init__(self, neurons, dt=1., tau_rc=100., v_th=-52., theta=0.05, rest=-65., tau_ref=5.):
        super().__init__()

        self.neurons = neurons

        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('v', torch.zeros(neurons))
        self.register_buffer('tau_rc', torch.tensor(tau_rc))
        self.register_buffer('v_th', torch.tensor(v_th))
        self.register_buffer('theta', torch.tensor(theta))
        self.register_buffer('s', torch.zeros(neurons))
        self.register_buffer('rest', torch.tensor(rest))
        self.register_buffer('refrac', torch.zeros(neurons))
        self.register_buffer('tau_ref', torch.tensor(tau_ref))

        self.v.fill_(self.rest)

    def reset(self):
        self.v[:] = self.rest

    def forward(self, x):
        self.v[:] = torch.exp(-self.dt / self.tau_rc) * (self.v - self.rest) + self.rest

        self.v += (self.refrac <= 0).float() * x

        self.refrac -= self.dt

        self.s[:] = self.v >= self.v_th + self.theta

        self.refrac.masked_fill_(self.s.bool(), self.tau_ref)
        self.v.masked_fill_(self.s.bool(), self.rest)

        return self.s, self.v


class DiehlAndCookPopulation(BasePopulation):
    def __init__(self,
                 neurons,
                 dt=1.,
                 tau_rc=100.,
                 v_th=-52.,
                 theta=0.05,
                 rest=-65.,
                 tau_ref=5.,
                 traces=True,
                 tau_tr=20.,
                 scale_tr=1.):
        super().__init__()

        self.neurons = neurons
        self.traces = traces

        self.register_buffer('dt', torch.tensor(dt))
        self.register_buffer('v', torch.zeros(neurons))
        self.register_buffer('tau_rc', torch.tensor(tau_rc))
        self.register_buffer('v_th', torch.tensor(v_th))
        self.register_buffer('theta', torch.tensor(theta))
        self.register_buffer('s', torch.zeros(neurons))
        self.register_buffer('rest', torch.tensor(rest))
        self.register_buffer('refrac', torch.zeros(neurons))
        self.register_buffer('tau_ref', torch.tensor(tau_ref))
        if traces:
            # TODO: 초기화는 어떤 값으로 해야 하는가?
            self.register_buffer('x', torch.zeros(neurons))
            self.register_buffer('tau_tr', torch.tensor(tau_tr))
            self.register_buffer('scale_tr', torch.tensor(scale_tr))

        self.reset()

    def reset(self):
        self.v.fill_(self.rest)
        self.x.fill_(0.)

    def forward(self, x):
        self.v[:] = torch.exp(-self.dt / self.tau_rc) * (self.v - self.rest) + self.rest

        self.v += (self.refrac <= 0).float() * x

        self.refrac -= self.dt

        self.s[:] = self.v >= self.v_th + self.theta

        self.refrac.masked_fill_(self.s.bool(), self.tau_ref)
        self.v.masked_fill_(self.s.bool(), self.rest)

        if self.traces:
            self.x[:] *= torch.exp(-self.dt / self.tau_tr)
            self.x[:] += self.scale_tr * self.s

        return self.s, self.v


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
