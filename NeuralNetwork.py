import numpy as np
import pandas as pd

from NeuralNetworkUtil import *
from random import uniform
from math import sqrt,fabs
from scipy.optimize import minimize

from decimal import Decimal

DEFAULT_DIRECTORY = './data/trainedData'

class NeuralNetwork:

    def __init__(self, numOfNodesPerLayer):
        # numOfNodesPerLayer is an array of integers giving the number of nodes per layer
        self.__construct(numOfNodesPerLayer)

    def predict(self, inputFeaturesVector, trainedParametersDirectory = DEFAULT_DIRECTORY):
        # this function performs forward propagation
        self.__loadTrainedParametersFromFiles(trainedParametersDirectory)
        targetPrediction = self.__feedForward(inputFeaturesVector)
        return targetPrediction[-1][1]

    def train(self, inputFeaturesVector, outputTargetsVector, regParam, trainedParametersDirectory = DEFAULT_DIRECTORY):
        # this function trains the neural network with backward propagation
        self.__initializeNeuralNetworkParameters()
        res = self.__calibrate(inputFeaturesVector, outputTargetsVector, regParam)
        print('Neural Network calibrate: ', res)
        self.__exportTrainedParametersToFiles(trainedParametersDirectory)
        return res

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
            theta = NeuralNetworkUtil.loadDataFromFile(trainedThetaFileName)
            if theta.shape.__len__() > 1:
                self.__thetas[counter] = theta
            else:
                self.__thetas[counter] = theta.reshape(1,theta.size)

    def __exportTrainedParametersToFiles(self,trainedParametersDirectory):
        # TODO check if the dimensions of the loaded parameters are the ones announced
        # in the construction of the neural network
        for counter,theta in zip(range(self.__numLayers-1), self.__thetas):
            trainedThetaFileName = trainedParametersDirectory + '/Theta' + str(counter) + '.txt'
            NeuralNetworkUtil.saveDataToFile(theta,trainedThetaFileName)

    def __feedForward(self, inputFeaturesVector):
        a = NeuralNetworkUtil.addBiasTerm(inputFeaturesVector)
        layerOutputs = list()
        layerOutputs.append(([],a))
        for layerIndex, theta in zip(range(self.__numLayers), self.__thetas):
            z = np.dot(a,theta.transpose())
            if layerIndex < self.__numLayers - 2:
                a = NeuralNetworkUtil.addBiasTerm(NeuralNetworkUtil.applyScalarFunction(z, NeuralNetworkUtil.sigmoid))
            else:
                a = NeuralNetworkUtil.applyScalarFunction(z, NeuralNetworkUtil.sigmoid)
            layerOutputs.append((z,a))
        return layerOutputs

    def __initializeNeuralNetworkParameters(self):
        numOfMatrices = self.__thetas.__len__()
        for index,theta in zip(range(numOfMatrices), self.__thetas):
            currentLayerSize = theta.shape[0]
            nextLayerSize = theta.shape[1]
            epsilon = sqrt(6.0) / sqrt(currentLayerSize + nextLayerSize)
            self.__thetas[index] = epsilon * (np.random.uniform(0,2.0,(currentLayerSize,nextLayerSize)) - 1.0)

    def __calibrate(self, inputFeaturesVector, outputTargetsVector, regParam):
        costFunction = lambda theta : self.__costFunction(theta, inputFeaturesVector, outputTargetsVector, regParam)
        optimizationOptions = {'maxiter': 600}
        thetaInit = NeuralNetworkUtil.roll(self.__thetas)
        res = minimize(fun = costFunction, x0 = thetaInit, method = 'L-BFGS-B', options = optimizationOptions, jac = True)
        # res = self.__GD(costFunction, thetaInit, {'maxiter': 400, 'learningRate': 5.0, 'tol': 1e-6})
        return res

    def __costFunction(self, vectorTheta, inputFeaturesVector, outputTargetsVector, regParam):
        # Initialise
        J = []
        m = inputFeaturesVector.__len__()
        # Cost function
        NeuralNetworkUtil.unroll(vectorTheta, self.__thetas)
        layerOutputs = self.__feedForward(inputFeaturesVector)
        J = NeuralNetworkUtil.logErrorClassificationFunction(layerOutputs[-1][1], outputTargetsVector)
        J += self.__addRegularizationParameter(m, regParam)
        # Cost function gradient
        flatGrad = []
        grads = self.__backPropagation(layerOutputs, outputTargetsVector, m, regParam)
        self.__addRegularizationParameterGrad(m, regParam, grads)
        flatGrad = NeuralNetworkUtil.roll(grads)
        return [J, flatGrad]

    def __backPropagation(self, layerOutputs, outputTargetsVector, m, regParam):
        # Initialization
        numObservations = outputTargetsVector.shape[0]
        numLayers = layerOutputs.__len__()
        grads = []

        # First iteration
        delta = layerOutputs[-1][1] - outputTargetsVector
        a2withBias = layerOutputs[-2][1]
        grad = np.dot(delta.transpose(), a2withBias) / numObservations
        grad += regParam / m * self.__thetas[-1]
        grads.append(grad)

        # Other layers
        for index, theta in reversed(list(enumerate(self.__thetas))):
            if index > 0:
                propError = np.dot(delta, theta)
                gZ = NeuralNetworkUtil.applyScalarFunction(layerOutputs[index][0], NeuralNetworkUtil.sigmoidGrad)
                delta = np.multiply(gZ,propError[:,1:])
                grad = np.dot(delta.transpose(),layerOutputs[index-1][1]) / numObservations
                grad += regParam / m * self.__thetas[index-1]
                grads.insert(0,grad)

        return grads

    def __addRegularizationParameter(self, m, regParam):
        Jreg = 0.0
        for theta in self.__thetas:
            Jreg += np.sum(theta[:,1:] * theta[:,1:])
        return 0.5 * regParam * Jreg / m

    def __addRegularizationParameterGrad(self, m, regParam, thetaGradList):
        numOfLayers = thetaGradList.__len__()
        for index, theta in zip(range(numOfLayers), self.__thetas):
            assert thetaGradList[index].shape == theta.shape
            thetaGradList[index][:,1:] += regParam * theta[:,1:] / m

    def __GD(self, costFunction, thetaInit, optimizationOptions):
        # This function performs a simple gradient descent
        learningRate = optimizationOptions['learningRate']
        maxiter = optimizationOptions['maxiter']
        tol = optimizationOptions['tol']
        vectorTheta = thetaInit

        res = {}
        res['success'] = False
        res['message'] = 'max iteration reached'

        l2GradDelta = 0.0

        for iterCounter in range(maxiter):
            [cost, grad] = costFunction(vectorTheta)
            vectorTheta -= learningRate * grad
            l2GradDelta = np.sum(grad * grad)
            if l2GradDelta < tol:
                res['message'] = 'optim successful'
                res['success'] = True
                break

        res['fun'] = cost
        res['funDelta'] = l2GradDelta
        res['numIter'] = iterCounter
        return res


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
