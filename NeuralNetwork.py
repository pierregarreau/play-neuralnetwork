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
        # TODO Export trained parameters to trainedParametersDirectory

    def getListOfThetas(self):
        return self.__thetas

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
        a = inputFeaturesVector
        layerOutputs = list()
        layerOutputs.append(([],a))
        for theta in self.__thetas:
            layerInputWithBias = NeuralNetworkUtil.addBiasTerm(a)
            z = np.dot(layerInputWithBias,theta.transpose())
            a = NeuralNetworkUtil.applyScalarFunction(z, NeuralNetworkUtil.sigmoid)
            layerOutputs.append((z,a))
        return layerOutputs

    def __initializeNeuralNetworkParameters(self):
        epsilon = 0.001
        for theta in self.__thetas:
            currentLayerSize = theta.shape[0]
            nextLayerSize = theta.shape[1]
            for inputIndex in range(currentLayerSize):
                for outputIndex in range(nextLayerSize):
                    theta[inputIndex, outputIndex] = uniform(-epsilon, epsilon)

    def __calibrate(self, inputFeaturesVector, outputTargetsVector):
        costFunction = lambda theta : self.__costFunction(theta, inputFeaturesVector, outputTargetsVector)
        optimizationOptions = {'gtol': 1e-6, 'disp': True}
        thetaInit = NeuralNetworkUtil.roll(self.__thetas)
        res = minimize(costFunction, thetaInit, method='BFGS', jac=True, options=optimizationOptions)
        pass

    def __costFunction(self, vectorTheta, inputFeaturesVector, outputTargetsVector):
        # Initialise
        J = []
        grad = []
        # Cost function
        NeuralNetworkUtil.unroll(vectorTheta, self.__thetas)
        layerOutputs = self.__feedForward(inputFeaturesVector)
        J = NeuralNetworkUtil.logErrorClassificationFunction(layerOutputs[-1][1], outputTargetsVector)
        # Cost function gradient
        grads = self.__backPropagation(layerOutputs, outputTargetsVector)
        flatGrad = NeuralNetworkUtil.roll(grads)
        return [J, flatGrad]

    def __backPropagation(self, layerOutputs, outputTargetsVector):
        # Initialization
        numObservations = outputTargetsVector.shape[0]
        numLayers = layerOutputs.__len__()
        grads = []

        # First iteration
        delta = layerOutputs[-1][1] - outputTargetsVector
        a2withBias = NeuralNetworkUtil.addBiasTerm(layerOutputs[1][1])
        grad = np.dot(delta.transpose(), a2withBias) / numObservations
        grads.append(grad)
        # Second layer
        propError = np.dot(delta, self.__thetas[-1])
        gZ = NeuralNetworkUtil.applyScalarFunction(layerOutputs[-2][0], NeuralNetworkUtil.sigmoidGrad)
        delta = np.multiply(gZ,propError[:,1:])
        grad = np.dot(delta.transpose(),NeuralNetworkUtil.addBiasTerm(layerOutputs[-3][1])) / numObservations
        grads.insert(0,grad)

        # TODO create loop to perform whatever the number of layers
        # for index, theta in reversed(list(enumerate(self.__thetas[1:]))):
        #     print(index)
        #     if index > 0:
        #         zNextLayer = layerOutputs[index][0] # remove assignment
        #         aCurrentLayer = layerOutputs[index][1] # remove assignment
        #
        #         deltaFactor = np.dot(delta,theta)
        #         deltaFactor = deltaFactor[:,1:]
        #
        #         delta = np.multiply(NeuralNetworkUtil.applyScalarFunction(zNextLayer, NeuralNetworkUtil.sigmoidGrad), deltaFactor)
        #
        #         grad = np.dot(aCurrentLayer.transpose(), delta) / numObservations
        #         grads.insert(grad,0)

        return grads

    # Below only used for computing numerial gradient (testing)
    def __feedForwardNoHistory(self, inputFeaturesVector):
        a = inputFeaturesVector
        for theta in self.__thetas:
            layerInputWithBias = NeuralNetworkUtil.addBiasTerm(a)
            z = np.dot(layerInputWithBias,theta.transpose())
            a = NeuralNetworkUtil.applyScalarFunction(z, NeuralNetworkUtil.sigmoid)
        return (z,a)

    def __costFunctionOnly(self, vectorTheta, inputFeaturesVector, outputTargetsVector):
        J = []
        NeuralNetworkUtil.unroll(vectorTheta, self.__thetas)
        layerOutputs = self.__feedForwardNoHistory(inputFeaturesVector)
        J = NeuralNetworkUtil.logErrorClassificationFunction(layerOutputs[1], outputTargetsVector)
        return J
