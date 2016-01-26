import numpy as np
import pandas as pd

from NeuralNetworkUtil import *
from random import uniform
from scipy.optimize import minimize

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
        self.__initializeNeuralNetworkParameters()
        self.__calibrate(inputFeaturesVector, outputTargetsVector)

    def __construct(self, numOfNodesPerLayer):
        if numOfNodesPerLayer.ndim > 1:
            print('Error : numOfNodesPerLayer needs to be a vector')
        else:
            self.__numLayers = numOfNodesPerLayer.size
            self.__thetas = [np.empty((nextLayer,currentLayer+1))
                for currentLayer, nextLayer in zip(numOfNodesPerLayer[:-1], numOfNodesPerLayer[1:])]

    def __loadTrainedParametersFromFiles(self,trainedParametersDirectory):
        # TODO check if the dimensions of the loaded parameters are the ones announced
        # in the construction of the neural network
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

    def __initializeNeuralNetworkParameters(self):
        epsilon = 0.001
        for theta in self.__thetas:
            currentLayerSize = theta.shape[0]
            nextLayerSize = theta.shape[1]
            for inputIndex in range(currentLayerSize):
                for outputIndex in range(nextLayerSize):
                    theta[inputIndex, outputIndex] = uniform(-epsilon, epsilon)

    def __calibrate(self, inputFeaturesVector, outputTargetsVector):
        # costFunction = lambda theta : self.__costFunction(theta, inputFeaturesVector, outputTargetsVector)
        self.__costFunction([], inputFeaturesVector, outputTargetsVector)
        # optimizationOptions = {'gtol': 1e-6, 'disp': True}
        thetaInit = NeuralNetworkUtil.roll(self.__thetas)
        # res = minimize(costFunction, thetaInit, method='BFGS', jac=True, options=optimizationOptions)

    def __costFunction(self, vectorTheta, inputFeaturesVector, outputTargetsVector):
        J = []
        grad = []
        NeuralNetworkUtil.unroll(vectorTheta, self.__thetas)
        targetPrediction = self.__feedForward(inputFeaturesVector)
        # J = NeuralNetworkUtil.logErrorFunction(targetPrediction, outputTargetsVector)

        # grads = self.__backPropagation(targetPrediction, outputTargetsVector)
        flatGrad = NeuralNetworkUtil.roll(grads)
        return [J, grad]

    def __backPropagation(self, targetPrediction, outputTargetsVector):
        #TODO
        pass
