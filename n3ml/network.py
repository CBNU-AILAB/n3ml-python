from typing import Dict

import torch
import torch.nn as nn

from n3ml.population import BasePopulation, InputPopulation
from n3ml.connection import BaseConnection


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.population = {}
        self.connection = {}

    def _add_population(self, name, population):
        self.population[name] = population

    def _add_connection(self, name, connection):
        self.connection[name] = connection

    def add_component(self, name, component):
        # Add a component to torch.nn.Module
        self.add_module(name, component)
        # Add a component to Network to handle some update strategies
        if isinstance(component, BasePopulation):
            self._add_population(name, component)
        elif isinstance(component, BaseConnection):
            self._add_connection(name, component)

    def init_param(self) -> None:
        for p in self.population.values():
            p.init_param()

    def normalize(self) -> None:
        for c in self.connection.values():
            c.normalize()

    def update(self) -> None:
        # TODO: update()를 어떻게 해야 추상화 할 수 있을까?
        # TODO: non-BP 기반 학습 알고리즘은 update()를 사용하여 학습을 수행한다.
        for c in self.connection.values():
            c.update()

    def run(self, x: Dict[str, torch.Tensor]) -> None:
        input = {}
        for name in x:
            input[name] = x[name].clone()
        for p_name in self.population:
            if isinstance(self.population[p_name], InputPopulation):
                if p_name not in input:
                    input[p_name] = torch.zeros(self.population[p_name].neurons)
            else:
                for c_name in self.connection:
                    if self.connection[c_name].target == self.population[p_name]:
                        if p_name in input:
                            input[p_name] += self.connection[c_name].run()
                        else:
                            input[p_name] = self.connection[c_name].run()

        for p_name in self.population:
            # print("{}'s input size: {}".format(p_name, input[p_name].size()))
            self.population[p_name].run(input[p_name])
        # print(input['exc'])
        # print(self.exc.v)


if __name__ == '__main__':
    pass
