import torch


class PoissonEncoder:
    # Poisson processor for encoding images
    def __init__(self, time_interval):
        self.time_interval = time_interval

    def __call__(self, images):
        # images.size: [b, c, h, w]
        # spiked_images.size: [t, b, c, h, w]
        b, c, h, w = images.size()
        r = images.unsqueeze(0).repeat(self.time_interval, 1, 1, 1, 1)
        p = torch.rand(self.time_interval, b, c, h, w)
        return (p <= r).float()


if __name__ == '__main__':
    r = torch.rand(1, 2, 2, 3)
    p = torch.rand(4, 1, 2, 2, 3)

    rr = r.unsqueeze(0).repeat(4, 1, 1, 1, 1)
    print(rr.size())

    print(rr)
    print(p)
    print(p <= rr)

    encoder = PoissonEncoder()

    print(encoder(r, 4))
