import torch


class BaseLearning:
    def __init__(self):
        pass


class ReSuMe(BaseLearning):
    def __init__(self):
        super().__init__()


class PostPre(BaseLearning):
    def __init__(self,
                 connection,
                 lr=(1e-4, 1e-2)):
        super().__init__()

        self.connection = connection
        # lr은 tuple 타입으로 크기가 2가 된다. lr[0]는 presynaptic spike에 대한 weight change에서 사용되고
        # lr[1]은 postsynaptic spike에 대한 weight change에서 사용된다.
        self.lr = lr

    def __call__(self):
        # Compute weight changes for presynaptic spikes
        s_pre = self.connection.source.s.unsqueeze(1)
        x_post = self.connection.target.x.unsqueeze(0)
        self.connection.w -= self.lr[0] * torch.transpose(torch.mm(s_pre, x_post), 0, 1)
        # Compute weight changes for postsynaptic spikes
        s_post = self.connection.target.s.unsqueeze(1)
        x_pre = self.connection.source.x.unsqueeze(0)
        self.connection.w += self.lr[1] * torch.mm(s_post, x_pre)
