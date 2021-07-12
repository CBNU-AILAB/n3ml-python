import torch
import torch.nn as nn

from n3ml.network import Network
from n3ml.population import InputPopulation, LIFPopulation, DiehlAndCookPopulation
from n3ml.connection import Connection
from n3ml.learning import ReSuMe, PostPre
from n3ml.layer import IF1d, IF2d, Conv2d, AvgPool2d, Linear, Bohte, TravanaeiAndMaida

import n3ml.layer
import n3ml.population
import n3ml.connection


class Wu2018(Network):
    def __init__(self,  batch_size, time_interval):
        super().__init__()
        self.conv1         = nn.Conv2d(1, 32,  kernel_size=3, stride=1, padding=1)
        self.conv2         = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1           = nn.Linear(7 * 7 * 32, 128)
        self.fc2           = nn.Linear(128, 10)
        self.avgpool       = nn.AvgPool2d(kernel_size=2)
        self.batch_size    = batch_size
        self.time_interval = time_interval

    def mem_update(self, ops, x, mem, spike):
        mem = mem * 0.2 * (1. - spike) + ops(x)
        STBP_spike = n3ml.layer.Wu()
        spike      = STBP_spike(mem)
        return mem, spike

    def forward(self, input):
        c1_mem = c1_spike = torch.zeros(self.batch_size, 32, 28, 28).cuda()
        c2_mem = c2_spike = torch.zeros(self.batch_size, 32, 14, 14).cuda()

        h1_mem = h1_spike               = torch.zeros(self.batch_size, 128).cuda()
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, 10).cuda()

        for time in range(self.time_interval):

            x = input > torch.rand(input.size()).cuda()

            c1_mem, c1_spike = self.mem_update(self.conv1, x.float(), c1_mem, c1_spike)
            x = self.avgpool(c1_spike)

            c2_mem, c2_spike = self.mem_update(self.conv2,x, c2_mem,c2_spike)
            x = self.avgpool(c2_spike)

            x = x.view(self.batch_size, -1)

            h1_mem, h1_spike = self.mem_update(self.fc1, x, h1_mem, h1_spike)
            h2_mem, h2_spike = self.mem_update(self.fc2, h1_spike, h2_mem,h2_spike)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / self.time_interval

        return outputs


class Ponulak2005(Network):
    def __init__(self,
                 neurons: int = 800,
                 num_classes: int = 10) -> None:
        super().__init__()
        self.neurons = neurons
        self.num_classes = num_classes
        self.add_component('input', n3ml.population.Input(1*28*28,
                                                          traces=False))
        self.add_component('hidden', n3ml.population.LIF(neurons,
                                                         tau_ref=2.0,
                                                         traces=False,
                                                         rest=0.0,
                                                         reset=0.0,
                                                         v_th=1.0,
                                                         tau_rc=10.0))
        self.add_component('output', n3ml.population.LIF(num_classes,
                                                         tau_ref=2.0,
                                                         traces=False,
                                                         rest=0.0,
                                                         reset=0.0,
                                                         v_th=1.0,
                                                         tau_rc=10.0))
        self.add_component('ih', n3ml.connection.Synapse(self.input, self.hidden))
        self.add_component('ho', n3ml.connection.Synapse(self.hidden, self.output))

    def reset_parameters(self):
        for synapse in self.connection.values():
            synapse.w[:] = torch.rand_like(synapse.w) - 0.5


class DiehlAndCook2015(Network):
    def __init__(self, neurons: int = 100):
        super().__init__()
        self.neurons = neurons
        self.add_component('inp', InputPopulation(1*28*28,
                                                  traces=True,
                                                  tau_tr=20.0))
        self.add_component('exc', DiehlAndCookPopulation(neurons,  #
                                                         traces=True,
                                                         rest=-65.0,
                                                         reset=-60.0,
                                                         v_th=-52.0,
                                                         tau_ref=5.0,
                                                         tau_rc=100.0,
                                                         tau_tr=20.0))
        self.add_component('inh', LIFPopulation(neurons,  #
                                                traces=False,
                                                rest=-60.0,
                                                reset=-45.0,
                                                v_th=-40.0,
                                                tau_rc=10.0,
                                                tau_ref=2.0,
                                                tau_tr=20.0))
        self.add_component('xe', Connection(self.inp,
                                            self.exc,
                                            mode='type0',
                                            learning=PostPre,
                                            w_min=0.0,
                                            w_max=1.0,
                                            norm=78.4))
        self.add_component('ei', Connection(self.exc,
                                            self.inh,
                                            mode='type2',
                                            w_min=0.0,
                                            w_max=22.5))
        self.add_component('ie', Connection(self.inh,
                                            self.exc,
                                            mode='type1',
                                            w_min=-120,
                                            w_max=0.0))


class Hunsberger2015(Network):
    def __init__(self, amplitude, tau_ref, tau_rc, gain, sigma, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            n3ml.layer.SoftLIF(amplitude=amplitude, tau_ref=tau_ref, tau_rc=tau_rc, gain=gain, sigma=sigma),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            n3ml.layer.SoftLIF(amplitude=amplitude, tau_ref=tau_ref, tau_rc=tau_rc, gain=gain, sigma=sigma),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=False),
            n3ml.layer.SoftLIF(amplitude=amplitude, tau_ref=tau_ref, tau_rc=tau_rc, gain=gain, sigma=sigma),
            nn.AvgPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 1024, bias=False),
            n3ml.layer.SoftLIF(amplitude=amplitude, tau_ref=tau_ref, tau_rc=tau_rc, gain=gain, sigma=sigma),
            nn.Linear(1024, self.num_classes, bias=False)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), 256)
        x = self.classifier(x)
        return x


class Bohte2002(Network):
    def __init__(self) -> None:
        super().__init__()

        self.add_component('fc1', Bohte(50, 10, time_constant=7.0))
        self.add_component('fc2', Bohte(10, 3, time_constant=7.0))
        # self.add_component('fc', Bohte(50, 3))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: 현재 시점
        # x: 스파이크 발화 시점에 대한 정보
        x = self.fc1(t, x)
        x = self.fc2(t, x)
        # x = self.fc(t, x)
        return x


class TravanaeiAndMaida2017(Network):
    def __init__(self,
                 num_classes: int = 10,
                 hidden_neurons: int = 100) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hidden_neurons = hidden_neurons
        self.add_component('fc1', TravanaeiAndMaida(in_neurons=1*28*28,
                                                    out_neurons=hidden_neurons,
                                                    threshold=0.9))
        self.add_component('fc2', TravanaeiAndMaida(in_neurons=hidden_neurons,
                                                    out_neurons=num_classes,
                                                    threshold=hidden_neurons*0.025))

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        o = self.fc1(o)
        o = self.fc2(o)
        return o

    def reset_variables(self, **kwargs):
        for l in self.layer.values():
            l.reset_variables(**kwargs)


class Cao2015_Tailored(Network):
    def __init__(self,
                 num_classes: int = 10,
                 in_planes: int = 3,
                 out_planes: int = 64) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.extractor = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(out_planes, out_planes, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(out_planes, out_planes, 3, bias=False),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_planes, out_planes, bias=False),
            nn.ReLU(),
            nn.Linear(out_planes, num_classes, bias=False)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Cao2015_SNN(Network):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Ho2013(Network):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TailoredCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, out_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.extractor = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(self.out_channels, self.out_channels, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(self.out_channels, self.out_channels, 3, bias=False),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels, bias=False),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.num_classes, bias=False)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class SpikingCNN(Network):
    def __init__(self, num_classes=10, in_channels=3, out_channels=64, time_interval=300):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_interval = time_interval

        self.extractor = nn.Sequential(
            Conv2d(self.in_channels, self.out_channels, 20, 20, 5, time_interval, bias=False),
            IF2d(self.out_channels, 20, 20, self.time_interval, threshold=5),
            AvgPool2d(2, 2, 64, 10, 10, self.time_interval),
            Conv2d(self.out_channels, self.out_channels, 6, 6, 5, time_interval, bias=False),
            IF2d(out_channels, 6, 6, self.time_interval, threshold=0.99),
            AvgPool2d(2, 2, 64, 3, 3, self.time_interval),
            Conv2d(self.out_channels, self.out_channels, 1, 1, 3, time_interval, bias=False),
            IF2d(out_channels, 1, 1, self.time_interval),
        )
        self.classifier = nn.Sequential(
            Linear(self.out_channels, self.out_channels, self.time_interval, bias=False),
            IF1d(self.out_channels, self.time_interval, threshold=0.99),
            Linear(self.out_channels, self.num_classes, self.time_interval, bias=False),
            IF1d(self.num_classes, self.time_interval)
        )

    def forward(self, t, o):
        for module in self.extractor:
            o = module(t, o)
        o = o.view(o.shape[0], o.shape[1], -1)
        for module in self.classifier:
            o = module(t, o)
        return o

    def add_layer(self, name, layer):
        self.add_module(name, layer)


if __name__ == '__main__':
    ann = TailoredCNN()
    print("The structure of standard CNN...")
    for _ in ann.named_children():
        print(_)
    print()

    snn = SpikingCNN()
    print("The structure of standard spiking CNN...")
    for _ in snn.named_children():
        print(_)
    print()

    snn.add_layer('added_linear', Linear(10, 10, 30, bias=False))
    snn.add_layer('added_IF1d', IF1d(10, 30))

    print("The structure of standard spiking CNN...")
    for _ in snn.named_children():
        print(_)
