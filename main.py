#! /usr/bin/env python
__author__ = 'PierreGarreau'

import matplotlib.pyplot as plt
import pandas as pd

from numpy import array,append,linspace,arange
from sklearn.cross_validation import train_test_split

from NeuralNetwork import *
from NeuralNetwork_test import initializeThetasForConfigB
from datadisplay import plot, plotCostFunction, plotCostFunctionVsRegParam, plotCostFunctionAfterTrainingVsRegParam
from NeuralNetworkUtil import *
from NeuralNetworkConfig import loadNNConfigA0, \
                                loadNNConfigA1, \
                                loadNNConfigB, \
                                loadNNConfigC, \
                                loadNNConfigD0, \
                                loadNNConfigD1

DATA_DIRECTORY = './data'

if __name__ == "__main__":
    # try:
        inputVectors,targets,neuralNetworkArchitecture = loadNNConfigD1()
        X_train, X_test, y_train, y_test = train_test_split(inputVectors, targets, train_size=0.85)

        nn = NeuralNetwork(neuralNetworkArchitecture)
        res = nn.train(X_train,y_train)

        predictions = nn.predict(X_train)
        predictionAccuracy = NeuralNetworkUtil.computePredictionAccuracy(predictions,y_train)
        print('The NN predicted the output with {}% accuracy'.format(predictionAccuracy*100))

        # Plotting routines Below
        # regParam = 0.1
        # plotCostFunction(regParam)
        # plotCostFunctionAfterTrainingVsRegParam()
        plot(inputVectors,100)

    # except ValueError:
        # print(ValueError)
