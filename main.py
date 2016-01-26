#! /usr/bin/env python
__author__ = 'PierreGarreau'

import NeuralNetwork as nn
import pandas as pd

from numpy import array
from NeuralNetworkUtil import *

DATA_DIRECTORY = './data'

if __name__ == "__main__":
    # try:
        inputVectorsFile = DATA_DIRECTORY + '/inputVectors.txt'
        inputVectors = NeuralNetworkUtil.loadDataFromFile(inputVectorsFile)

        targetsFile = DATA_DIRECTORY + '/targetVectors.txt'
        targets = NeuralNetworkUtil.loadDataFromFile(targetsFile)

        neuralNetworkArchitecture = array([400,25,10])
        xNor = nn.NeuralNetwork(neuralNetworkArchitecture)

        xNor.train(inputVectors,targets)

        # predictedTargets = xNor.predict(inputVectors)
        # print(NeuralNetworkUtil.computePredictionAccuracy(predictedTargets, targets))
    # except ValueError:
        # print(ValueError)
