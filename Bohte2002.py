import argparse

import torch

import n3ml.model


def app(opt):
    model = n3ml.model.Bohte2002()

    x = (torch.rand(50) * 30).int()
    # x[10:20] = -1
    x[:30] = -1

    model.initialize(delay=True)

    # print(model.fc1.s)
    # print(model.fc2.s)
    print(x)
    print(model.fc1.s)
    print(model.fc2.s)
    print()

    for t in range(60):
        model(torch.tensor(t).float(), x)
        print("t: {}".format(t))
        print(model.fc1.s)
        print(model.fc2.s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    app(parser.parse_args())
