import torch

import matplotlib.pyplot as plt


if __name__ == '__main__':
    w = torch.rand((100, 28, 28))
    w[:25] = torch.rand(25, 28, 28) * 0.5
    w[75:] = torch.rand(25, 28, 28) * 0.2
    rows = 10
    cols = 10
    ww = torch.rand((280, 280))

    for r in range(rows):
        for c in range(cols):
            ww[r*28:(r+1)*28, c*28:(c+1)*28] = w[r*10+c]

    plt.matshow(ww)

    plt.show()
