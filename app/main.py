#! /usr/bin/env python
__author__ = 'PierreGarreau'

import matplotlib.pyplot as plt

import numpy as np
from sklearn.cross_validation import train_test_split

from analytics.NeuralNetwork import *
from analytics.NeuralNetworkUtil import *
from util.datadisplay import plot, plotCostFunction, plotCostFunctionVsRegParam, plotCostFunctionAfterTrainingVsRegParam
from data.load import Load

DATA_DIRECTORY = './data'

if __name__ == "__main__":
    # try:
    features, labels = Load.mnist(10)
    neuralNetworkArchitecture = np.array([400, 25, 10])
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, train_size=0.85)
    print(features.shape, labels.shape, neuralNetworkArchitecture)
    nn = NeuralNetwork(neuralNetworkArchitecture)
    res = nn.train(X_train, y_train)

    predictions = nn.predict(X_train)
    predictionAccuracy = NeuralNetworkUtil.computePredictionAccuracy(
        predictions, y_train)
    print('The NN predicted the output with {}% accuracy'.format(
        predictionAccuracy * 100))

    # Plotting routines Below
    # regParam = 0.1
    # plotCostFunction(regParam)
    # plotCostFunctionAfterTrainingVsRegParam()
    # plot(features,100)

# except ValueError:
# print(ValueError)
