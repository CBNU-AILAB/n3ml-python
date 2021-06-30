import torch
import torch.nn as nn

from n3ml.population import BasePopulation


class BaseConnection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def init_param(self) -> None:
        # mode: type0 - many-to-many
        #       type1 - many-to-many w/o same poisiton
        #       type2 - one-to-one
        # 이 구현에서는 source와 target의 크기가 서로 같다고 가정한다.
        # 확장을 하기 위해서는 서로 다른 크기에 대해서도 대각행렬을 만들 수 있는
        # 구현을 해야 한다.
        if self.mode == 'type0':
            # m2m
            self.w[:] = torch.rand(self.w.size()) * 0.3
        elif self.mode == 'type1':
            # m2m w/o same position
            # 이 시냅스는 inhibitory syanpse로 사용되기 때문에 negative로 초기화한다.
            # 이 시냅스와 별개로 negative로 설정할 시냅스를 입력받아 처리하도록 확장해야 한다.
            self.w[:] = (torch.ones(self.w.size()) * -120).fill_diagonal_(0.0)
        elif self.mode == 'type2':
            # o2o
            self.w[:] = torch.diagflat(torch.ones(self.w.size()[0]) * 22.5)


class Connection(BaseConnection):
    # This Connection class is designed to connect between two 1D Population instances
    def __init__(self,
                 source: BasePopulation,
                 target: BasePopulation,
                 learning=None,
                 mode='type0',
                 w_min: float = 0.0,
                 w_max: float = 1.0,
                 norm: float=None):
        super().__init__()
        # Now, both source and target must be 1D Population instances
        self.source = source
        self.target = target
        # learning must take a callable as an argument
        if learning:
            self.learning = learning(self)
        else:
            self.learning = learning
        # mode: type0
        #       type1
        #       type2
        self.mode = mode
        self.register_buffer('w', torch.zeros((target.neurons, source.neurons)))
        self.w_min = w_min
        self.w_max = w_max
        self.norm = norm

        self.init_param()

    def normalize(self) -> None:
        if self.norm is not None:
            w_abs_sum = self.w.abs().sum(1).unsqueeze(1)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.w *= self.norm / w_abs_sum

    def run(self) -> torch.Tensor:
        # run() 함수는 presynaptic population에서 발화된 스파이크와
        # 시냅스 가중치 간에 계산된 결과를 출력한다.
        return torch.matmul(self.w, self.source.s)

    def update(self):
        if self.learning:
            self.learning()
