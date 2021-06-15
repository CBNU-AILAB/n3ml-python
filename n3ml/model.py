import torch.nn as nn

from n3ml.network import Network
from n3ml.population import InputPopulation, LIFPopulation, DiehlAndCookPopulation
from n3ml.connection import Connection
from n3ml.learning import ReSuMe, PostPre
from n3ml.layer import IF1d, IF2d, Conv2d, AvgPool2d, Linear


class Ponulak2005(Network):
    def __init__(self):
        super().__init__()
        self.add_component('input', InputPopulation(1*28*28))
        self.add_component('hidden', LIFPopulation(100))
        self.add_component('output', LIFPopulation(10))
        self.add_component('ih', Connection(self.input, self.hidden))
        self.add_component('ho', Connection(self.hidden, self.output, learning=ReSuMe))

    def reset(self):
        self.hidden.reset()
        self.output.reset()

    def forward(self, x):
        x = self.input(x)
        x = self.ih(x)
        x, _ = self.hidden(x)
        x = self.ho(x)
        x, _ = self.output(x)
        return x


class DiehlAndCook2015(Network):
    def __init__(self):
        super().__init__()
        self.add_component('inp', InputPopulation(1*28*28))
        self.add_component('exc', DiehlAndCookPopulation(100))
        self.add_component('inh', LIFPopulation(100))
        self.add_component('xe', Connection(self.inp, self.exc, learning=PostPre))
        self.add_component('ei', Connection(self.exc, self.inh))
        self.add_component('ie', Connection(self.inh, self.exc))

    def reset(self):
        self.inp.reset()
        self.exc.reset()
        self.inh.reset()

    def update(self):
        # TODO: update()를 어떻게 해야 추상화 할 수 있을까?
        # TODO: non-BP 기반 학습 알고리즘은 update()를 사용하여 학습을 수행한다.
        self.xe.update()

    def forward(self, x):
        x = self.inp(x)
        x = self.xe(x)
        x, _ = self.exc(x)
        x = self.ei(x)
        x, _ = self.inh(x)
        x = self.ie(x)
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
