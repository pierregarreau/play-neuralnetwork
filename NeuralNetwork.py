import numpy as np
import pandas as pd

from NeuralNetworkUtil import *

DEFAULT_DIRECTORY = './data/trainedData'

class NeuralNetwork:

    def __init__(self, numOfNodesPerLayer):
        # numOfNodesPerLayer is an array of integers giving the number of nodes per layer
        self.__construct(numOfNodesPerLayer)

    def predict(self, inputFeaturesVector, trainedParametersDirectory = DEFAULT_DIRECTORY):
        # this function performs forward propagation
        self.__loadTrainedParametersFromFiles(trainedParametersDirectory)
        targetPrediction = self.__feedForward(inputFeaturesVector)
        return targetPrediction

    def train(self, inputFeaturesVector, outputTargetsVector, trainedParametersDirectory = DEFAULT_DIRECTORY):
        # this function trains the neural network with backward propagation
        pass

    def __construct(self, numOfNodesPerLayer):
        if numOfNodesPerLayer.ndim > 1:
            print('Error : numOfNodesPerLayer needs to be a vector')
        else:
            self.__numLayers = numOfNodesPerLayer.size
            self.__thetas = [np.empty(currentLayer,nextLayer)
                for currentLayer, nextLayer in zip(numOfNodesPerLayer[:-1], numOfNodesPerLayer[1:])]

    def __loadTrainedParametersFromFiles(self,trainedParametersDirectory):
        for counter in range(self.__numLayers-1):
            trainedThetaFileName = trainedParametersDirectory + '/Theta' + str(counter) + '.txt'
            self.__thetas[counter] = NeuralNetworkUtil.loadDataFromFile(trainedThetaFileName)

    def __feedForward(self, inputFeaturesVector):
        layerOutput = inputFeaturesVector
        for theta in self.__thetas:
            layerInputWithBias = NeuralNetworkUtil.addBiasTerm(layerOutput)
            layerOutput = np.dot(layerInputWithBias,theta.transpose())
            NeuralNetworkUtil.applyScalarFunction(layerOutput,NeuralNetworkUtil.sigmoid)
        return layerOutput
