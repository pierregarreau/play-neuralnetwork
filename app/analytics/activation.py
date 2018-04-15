import numpy as np

from typing import Union
from abc import ABCMeta, abstractclassmethod


class Activation(metaclass=ABCMeta):
    @staticmethod
    @abstractclassmethod
    def function(z: np.array) -> np.array:
        pass

    @staticmethod
    @abstractclassmethod
    def derivative(z: np.array) -> np.array:
        pass


class Sigmoid(Activation):
    @staticmethod
    def function(z: np.array) -> np.array:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def derivative(z: np.array) -> np.array:
        return Sigmoid.function(z) * (1.0 - Sigmoid.function(z))


class ActivationFactory():
    def create(type) -> Union[Activation, bool]:
        if type == 'sigmoid':
            return Sigmoid()
        raise ValueError('bad activation function type')
        return False
