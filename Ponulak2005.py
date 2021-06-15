import argparse

import torch
import torchvision

from n3ml.encoder import PoissonEncoder
from n3ml.model import Ponulak2005

import numpy as np
import matplotlib.pyplot as plt


def rasterplot(time, spikes, **kwargs):
    ax = plt.gca()

    n_spike, n_neuron = spikes.shape

    kwargs.setdefault("linestyle", "None")
    kwargs.setdefault("marker", "|")

    spiketimes = []

    for i in range(n_neuron):
        temp = time[spikes[:, i] > 0].ravel()
        spiketimes.append(temp)

    spiketimes = np.array(spiketimes)

    indexes = np.zeros(n_neuron, dtype=np.int)

    for t in range(time.shape[0]):
        for i in range(spiketimes.shape[0]):
            if spiketimes[i].shape[0] <= 0:
                continue
            if indexes[i] < spiketimes[i].shape[0] and \
                    time[t] == spiketimes[i][indexes[i]]:
                ax.plot(spiketimes[i][indexes[i]], i + 1, 'k', **kwargs)

                plt.draw()
                plt.pause(0.002)

                indexes[i] += 1

    plt.close('all')


def _plot(observer):
    # 시각화 하고자 하는 것이 무엇이냐에 따라 형식이 다를 수 있다.
    # 그러면, 처음 입력에서 타입을 먼저 식별할 필요가 있다.
    # 지금은 observer 내에 있는 모든 것들의 타입이 spike train이라고 가정한다.

    # n이 너무 크면 행렬로 만드는 방법도 필요하다.
    n = len(observer)

    fig, axs = plt.subplots(n)

    return


def plot(spikes):
    spikes = np.stack(spikes)
    time_interval = spikes.shape[0]

    times = np.arange(0, time_interval)

    rasterplot(times, spikes)


def app(opt):
    print(opt)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=True)

    input_encoder = PoissonEncoder()

    model = Ponulak2005()

    for epoch in range(opt.num_epochs):
        for images, labels in train_loader:
            model.reset()

            spiked_images = input_encoder(images, opt.time_interval)    # [time_interval, b, c, h, w]
            spiked_images = spiked_images.view(opt.time_interval, 1, -1)   # [time_interval, b, c*h*w]

            # TODO: 시각화 도구를 어떻게 사용해야 할까?
            observer = {
                'input.s': [],
                'hidden.s': [],
                'output.s': []
            }

            for time in range(opt.time_interval):
                model(spiked_images[time])

                print(model.output.s.numpy())
                # TODO: 자동으로 numpy로 변환하는 부분이 필요하다.
                # observer['input.s'].append(model.input.s.numpy())
                # observer['hidden.s'].append(model.hidden.s.numpy())

            # plot(observer['hidden.s'])
            exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--time_step', default=1, type=int)         # 1ms
    parser.add_argument('--time_interval', default=50, type=int)    # 50ms

    parser.add_argument('--num_epochs', default=30, type=int)

    app(parser.parse_args())
