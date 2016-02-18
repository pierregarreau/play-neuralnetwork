#! /usr/bin/env python
__author__ = 'PierreGarreau'

from NeuralNetwork import *
import pandas as pd

from numpy import array
from NeuralNetworkUtil import *
from NeuralNetworkConfig import loadNNConfigC

DATA_DIRECTORY = './data'

if __name__ == "__main__":
    # try:
        inputVectors,targets,neuralNetworkArchitecture = loadNNConfigC()
        nn = NeuralNetwork(neuralNetworkArchitecture)
        nn.train(inputVectors,targets)
        inputFeaturesVector = array([[0,0],[0,1],[1,0],[1,1]])
        predictions = nn.predict(inputVectors)
        # computePredictionAccuracy
        targetValues = NeuralNetworkUtil.transformClassificationTargetToValue(targets)
        predictionValues = NeuralNetworkUtil.transformClassificationTargetToValue(predictions)
        predictionAccuracy = NeuralNetworkUtil.computePredictionAccuracy(predictions,targetValues)
        print('The NN predicted the output with {}% accuracy'.format(predictionAccuracy*100))
        print(zip(predictionValues,targetValues)[:10])
        # predictedTargets = xNor.predict(inputVectors)
        # print(NeuralNetworkUtil.computePredictionAccuracy(predictedTargets, targets))
    # except ValueError:
        # print(ValueError)
