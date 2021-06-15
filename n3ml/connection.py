import torch
import torch.nn as nn


class BaseConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def init_param(self):
        # Initialize the learnable parameters in a network
        self.w[:] = torch.rand(self.w.size()) * 0.01


class Connection(BaseConnection):
    # This Connection class is designed to connect between two 1D Population instances
    def __init__(self, source, target, learning=None, mode='m2m'):
        super().__init__()
        # Now, both source and target must be 1D Population instances
        self.source = source
        self.target = target
        # learning must take a callable as an argument
        if learning:
            self.learning = learning(self)
        else:
            self.learning = learning
        # mode: m2m (many-to-many)
        #       o2o (one-to-one)
        #       m2o (many-to-one)
        #       o2m (one-to-many)
        self.mode = mode

        if mode == 'm2m':
            self.register_buffer('w', torch.zeros((target.neurons, source.neurons)))
        elif mode == 'o2o':
            pass
        elif mode == 'm2o':
            pass
        elif mode == 'o2m':
            pass

        self.init_param()

    def forward(self, x):
        return torch.matmul(self.w, x)

    def update(self):
        if self.learning:
            self.learning()
