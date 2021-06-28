import argparse

import torch
import torchvision

from n3ml.encoder import PoissonEncoder
from n3ml.model import DiehlAndCook2015
from n3ml.visualizer import plot_w


def app(opt):
    # Load MNIST dataset
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            opt.data,
            train=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
        batch_size=opt.batch_size,
        shuffle=True)

    # Define an encoder to generate spike train for an image
    encoder = PoissonEncoder(opt.time_interval)

    # Define a model
    model = DiehlAndCook2015()

    fig = None
    mat = None

    # Conduct training phase
    for epoch in range(opt.num_epochs):
        for images, labels in train_loader:
            # Initialize a model
            model.init_param()
            # Encode images into spiked_images
            spiked_images = encoder(images)
            spiked_images = spiked_images.view(opt.time_interval, opt.batch_size, -1)
            # Train a model
            for time in range(opt.time_interval):
                model.run({'inp': spiked_images[time]})
                # print(model.exc.v_th + model.exc.theta)
                # print(model.inp.s)
                # print(model.exc.s.sum())
            # Update weights using learning rule
            model.update()

            w = model.xe.w.detach().cpu().numpy()
            fig, mat = plot_w(fig, mat, w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--time_step', default=1, type=int)         # 1ms
    parser.add_argument('--time_interval', default=250, type=int)   # 250ms

    parser.add_argument('--num_epochs', default=30, type=int)

    app(parser.parse_args())