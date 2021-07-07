import time
import argparse

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import n3ml.model
import n3ml.encoder
import n3ml.optimizer

np.set_printoptions(threshold=np.inf, linewidth=np.nan)


def validate(loader, model, encoder):
    pass


def label_encoder(label, beta, num_classes, time_interval):
    """
        1초 동안에 생성될 수 있는 spikes의 수는 얼마나 될까?
        이는 continuous time domain인지 discrete time domain인지에 따라 달라질 수 있다.
        예를 들어, continuous의 경우는 최대로 생성될 수 있는 스파이크 수는 무한대가 된다.
        반면에 discrete의 경우는 최대로 생성될 수 있는 스파이크 수는 time step에 영향을 받는다.

        현재 구현에서는 time step이 1ms라고 가정을 하고 spike train을 생성한다.
        향후에 입력된 time step에 따라 적절한 spike train을 생성하는 방법을 구현할 필요가 있다.
    """
    r = torch.zeros((time_interval, num_classes))
    r[:, label] = torch.rand(time_interval) <= (beta/1000)
    return r


def train(loader, model, encoder, optimizer, opt):
    for image, label in loader:
        # Squeeze batch dimension
        # Now, batch processing isn't supported
        image = image.squeeze(dim=0)
        label = label.squeeze()

        spiked_image = encoder(image)
        spiked_image = spiked_image.view(spiked_image.size(0), -1)

        spiked_label = label_encoder(label, opt.beta, opt.num_classes, opt.time_interval)

        # print(label)
        # print(spiked_label)
        # exit(0)

        # np_spiked_image = spiked_image.numpy()

        spike_buffer = {
            'inp': [],
            'fc1': [],
            'fc2': []
        }

        print()
        print("label: {}".format(label))

        for t in range(opt.time_interval):
            # print(np_spiked_image[t])

            model(spiked_image[t])

            spike_buffer['inp'].append(spiked_image[t].clone())
            spike_buffer['fc1'].append(model.fc1.o.clone())
            spike_buffer['fc2'].append(model.fc2.o.clone())

            for l in spike_buffer.values():
                if len(l) > 5:
                    l.pop(0)

            # print(model.fc1.u.numpy())
            # print(model.fc1.o.numpy())
            # print(model.fc2.u.numpy())
            print(model.fc2.o.numpy())

            # time.sleep(1)

            optimizer.step(spike_buffer, spiked_label[t], label)

        model.reset_variables(w=False)


def app(opt):
    print(opt)

    # Load MNIST
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=False,
            transform=torchvision.transforms.Compose([transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=False)

    # Make a model
    model = n3ml.model.TravanaeiAndMaida2017(opt.num_classes, hidden_neurons=opt.hidden_neurons)
    model.reset_variables()

    # Make an encoder
    encoder = n3ml.encoder.Simple(time_interval=opt.time_interval)

    # Make an optimizer
    optimizer = n3ml.optimizer.TavanaeiAndMaida(model, lr=opt.lr)

    for epoch in range(opt.num_epochs):
        train(train_loader, model, encoder, optimizer, opt)

        validate(val_loader, model, encoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--num_epochs', default=120, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--time_interval', default=200, type=int)
    parser.add_argument('--beta', default=250, type=float)            # 250 Hz
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--hidden_neurons', default=500, type=int)

    app(parser.parse_args())
