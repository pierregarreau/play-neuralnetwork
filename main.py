#! /usr/bin/env python
__author__ = 'PierreGarreau'

import NeuralNetwork as nn
import pandas as pd

from numpy import array
from NeuralNetworkUtil import *
from NeuralNetworkConfig import loadNNConfigB

DATA_DIRECTORY = './data'

if __name__ == "__main__":
    # try:
        inputVectors,targets,neuralNetworkArchitecture = loadNNConfigB()
        xNor = nn.NeuralNetwork(neuralNetworkArchitecture)
        xNor.train(inputVectors,targets)

        # predictedTargets = xNor.predict(inputVectors)
        # print(NeuralNetworkUtil.computePredictionAccuracy(predictedTargets, targets))
    # except ValueError:
        # print(ValueError)
