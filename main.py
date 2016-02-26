#! /usr/bin/env python
__author__ = 'PierreGarreau'

from NeuralNetwork import *
from NeuralNetwork_test import initializeThetasForConfigB
import pandas as pd

from numpy import array
from NeuralNetworkUtil import *
from NeuralNetworkConfig import loadNNConfigA0, \
                                loadNNConfigA1, \
                                loadNNConfigB, \
                                loadNNConfigC, \
                                loadNNConfigD0

DATA_DIRECTORY = './data'

if __name__ == "__main__":
    # try:
        inputVectors,targets,neuralNetworkArchitecture = loadNNConfigB()
        nn = NeuralNetwork(neuralNetworkArchitecture)
        nn.train(inputVectors,targets)
        # initializeThetasForConfigB(nn)
        predictions = nn.predict(inputVectors)
        # computePredictionAccuracy
        predictionAccuracy = NeuralNetworkUtil.computePredictionAccuracy(predictions,targets)
        print('The NN predicted the output with {}% accuracy'.format(predictionAccuracy*100))
        print('The trained parameters are: ', nn.getListOfThetas())
        # print(zip(predictionValues,targetValues)[:10])
        # print(array(zip(inputVectors[:10],targets,predictions[:10])))
        # predictedTargets = xNor.predict(inputVectors)
        # print(NeuralNetworkUtil.computePredictionAccuracy(predictedTargets, targets))
    # except ValueError:
        # print(ValueError)
