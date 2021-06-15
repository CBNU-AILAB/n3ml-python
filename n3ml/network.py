import torch.nn as nn

from n3ml.population import BasePopulation
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


if __name__ == '__main__':
    pass
