#! /usr/bin/env python
__author__ = 'PierreGarreau'

import NeuralNetwork as nn
from numpy import array

if __name__ == "__main__":
    try:
        neuralNetworkArchitecture = array([2,2,1])
        xNor = nn.NeuralNetwork(neuralNetworkArchitecture)
    except ValueError:
        print(ValueError)
