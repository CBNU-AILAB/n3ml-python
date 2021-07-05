"""
    Bohte2002 example.

    Now, spikeprop algorithm can be applied to only feed-forward neural network.
    It means that when we construct neural network only consider a sequential structure.
    We achieve this using nn.Sequential.
"""
import argparse

import torch

import n3ml.model
import n3ml.data
import n3ml.encoder
import n3ml.optimizer


class LabelEncoder:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def run(self, label):
        o = torch.zeros(self.num_classes)
        o.fill_(13)  # 15
        o[label].fill_(7)  # 5
        return o


def rmse(pred, target):
    # print("pred: {} - target: {}".format(pred, target))
    return torch.sum((pred[pred >= 0]-target[pred >= 0])**2)/2


def do_correct(o, y):
    if torch.argmin(o) == (o.size(0)-torch.argmin(torch.flip(o, [0]))-1):
        return torch.argmin(o) == torch.argmin(y)
    return 0


def validate(data, model, data_encoder, label_encoder, opt):
    total_data = 0
    corrects = 0

    for i in range(data['test.data'].size(0)):
        model.initialize(delay=False)

        input = data['test.data'][i]
        label = data['test.target'][i]

        spiked_input = data_encoder.run(input)
        spiked_input = torch.cat((spiked_input.view(-1), torch.zeros(2)))
        spiked_label = label_encoder.run(label)

        for t in range(opt.num_steps):
            model(torch.tensor(t).float(), spiked_input)
        o = model.fc2.s

        total_data += 1
        corrects += do_correct(o, spiked_label)

    return corrects.float() / total_data


def train(data, model, data_encoder, label_encoder, optimizer, loss, epoch, opt):
    total_data = 0
    corrects = 0

    for i in range(data['train.data'].size(0)):
        model.initialize(delay=False)

        input = data['train.data'][i]
        label = data['train.target'][i]

        spiked_input = data_encoder.run(input)
        spiked_input = torch.cat((spiked_input.view(-1), torch.zeros(2)))
        spiked_label = label_encoder.run(label)

        for t in range(opt.num_steps):
            model(torch.tensor(t).float(), spiked_input)
        o = model.fc2.s

        print(model.fc1.s)
        print(model.fc2.s)
        print("pred: {} - target: {}".format(o, spiked_label))
        # l = loss(o, spiked_label)
        # print("loss: {}".format(l))

        optimizer.step(model, spiked_input, spiked_label, epoch)

        total_data += 1
        corrects += do_correct(o, spiked_label)

    return corrects.float() / total_data


def app(opt):
    import numpy as np
    np.set_printoptions(threshold=np.inf)

    print(opt)

    data_loader = n3ml.data.IRISDataLoader()
    data = data_loader.run()
    summary = data_loader.summarize()

    data_encoder = n3ml.encoder.Population(neurons=12,
                                           minimum=summary['min'],
                                           maximum=summary['max'],
                                           max_firing_time=opt.max_firing_time,
                                           not_to_fire=opt.not_to_fire,
                                           dt=opt.dt)
    label_encoder = LabelEncoder(opt.num_classes)

    model = n3ml.model.Bohte2002()
    model.initialize()

    optimizer = n3ml.optimizer.Bohte()

    for epoch in range(opt.num_epochs):
        acc = train(data, model, data_encoder, label_encoder, optimizer, rmse, epoch, opt)
        print("epoch: {} - tr. acc: {}".format(epoch, acc))

        acc = validate(data, model, data_encoder, label_encoder, opt)
        print("epoch: {} - val. acc: {}".format(epoch, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_epochs', default=90, type=int)
    parser.add_argument('--dt', default=1, type=int)
    parser.add_argument('--num_steps', default=40, type=int)
    parser.add_argument('--max_firing_time', default=30, type=int)
    parser.add_argument('--not_to_fire', default=28, type=int)

    app(parser.parse_args())
