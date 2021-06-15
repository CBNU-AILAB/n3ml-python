import torch
import torchvision


class BaseDataLoader:
    def __init__(self):
        pass


class MNISTDataLoader(BaseDataLoader):
    def __init__(self, data, batch_size=1, shuffle=True, train=True):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train
        self.loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                data,
                train=train,
                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
            batch_size=batch_size,
            shuffle=shuffle)
        self.it = iter(self.loader)
        self.o = next(self.it)

    def next(self):
        self.o = next(self.it)

    def __call__(self):
        return self.o


class CIFAR10DataLoader(BaseDataLoader):
    def __init__(self):
        pass
