import numpy as np

from random import uniform
from math import sqrt, fabs, floor
from scipy.optimize import minimize
from random import sample
from decimal import Decimal

from analytics.NeuralNetworkUtil import *
from data.load import Load

DEFAULT_DIRECTORY = './data/trainedData'


class NeuralNetwork:
    def __init__(self, numOfNodesPerLayer):
        # numOfNodesPerLayer is an array of integers giving the number of nodes per layer
        self.__construct(numOfNodesPerLayer)

    def predict(self,
                inputFeaturesVector,
                trainedParametersDirectory=DEFAULT_DIRECTORY):
        # this function performs forward propagation
        # TODO why is the load from file below needed?
        # self.__loadTrainedParametersFromFiles(trainedParametersDirectory)
        targetPrediction = self.__feedForward(inputFeaturesVector)
        return targetPrediction[-1][1]

    def train(self,
              inputFeaturesVector,
              outputTargetsVector,
              regParam=0.1,
              trainedParametersDirectory=DEFAULT_DIRECTORY):
        # this function trains the neural network with backward propagation
        self.__initializeNeuralNetworkParameters()
        res = self.__calibrate(inputFeaturesVector, outputTargetsVector,
                               regParam)
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
            self.__thetas = [
                np.empty((nextLayer, currentLayer + 1))
                for currentLayer, nextLayer in zip(numOfNodesPerLayer[:-1],
                                                   numOfNodesPerLayer[1:])
            ]

    def __loadTrainedParametersFromFiles(self, trainedParametersDirectory):
        # TODO check if the dimensions of the loaded parameters are the ones announced
        # in the construction of the neural network
        for counter in range(self.__numLayers - 1):
            trainedThetaFileName = trainedParametersDirectory + '/Theta' + str(
                counter) + '.txt'
            theta = Load.from_file(trainedThetaFileName)
            if theta.shape.__len__() > 1:
                self.__thetas[counter] = theta
            else:
                self.__thetas[counter] = theta.reshape(1, theta.size)

    def __exportTrainedParametersToFiles(self, trainedParametersDirectory):
        # TODO check if the dimensions of the loaded parameters are the ones announced
        # in the construction of the neural network
        for counter, theta in zip(range(self.__numLayers - 1), self.__thetas):
            trainedThetaFileName = trainedParametersDirectory + '/Theta' + str(
                counter) + '.txt'
            Load.to_file(theta, trainedThetaFileName)

    def __feedForward(self, inputFeaturesVector):
        a = NeuralNetworkUtil.addBiasTerm(inputFeaturesVector)
        layerOutputs = list()
        layerOutputs.append(([], a))
        for layerIndex, theta in zip(range(self.__numLayers), self.__thetas):
            z = np.dot(a, theta.transpose())
            # TODO refactor this sigmoid into function
            sigmoid = 1.0 / (1.0 + np.exp(-z))
            if layerIndex < self.__numLayers - 2:
                a = NeuralNetworkUtil.addBiasTerm(sigmoid)
            else:
                a = sigmoid
            layerOutputs.append((z, a))
        return layerOutputs

    def __initializeNeuralNetworkParameters(self):
        numOfMatrices = self.__thetas.__len__()
        for index, theta in zip(range(numOfMatrices), self.__thetas):
            currentLayerSize = theta.shape[0]
            nextLayerSize = theta.shape[1]
            epsilon = sqrt(6.0) / sqrt(currentLayerSize + nextLayerSize)
            self.__thetas[index] = epsilon * (
                np.random.uniform(0, 2.0,
                                  (currentLayerSize, nextLayerSize)) - 1.0)

    def __calibrate(self, inputFeaturesVector, outputTargetsVector, regParam):
        costFunction = lambda theta, inputFeaturesVector, outputTargetsVector : self.__costFunction(theta, inputFeaturesVector, outputTargetsVector, regParam)
        thetaInit = NeuralNetworkUtil.roll(self.__thetas)
        minimizationOptions = {'optimizer': '', 'maxiter': 1000, 'tol': 1e-7}
        res = self.__minimize(costFunction, thetaInit, minimizationOptions, inputFeaturesVector, outputTargetsVector)
        return res

    def __costFunction(self, vectorTheta, inputFeaturesVector, outputTargetsVector, regParam = 0.1):
        # costFunction returns the objective function and its gradient
        # computed for the input feature vector and the targets. Optional parameters
        # are the refularization parameters which makes the objective function more
        # convex and the index which, if not empty, is used to reduce the training
        # set.
        J = []
        m = inputFeaturesVector.__len__()
        NeuralNetworkUtil.unroll(vectorTheta, self.__thetas)
        layerOutputs = self.__feedForward(inputFeaturesVector)
        J = NeuralNetworkUtil.logErrorClassificationFunction(
            layerOutputs[-1][1], outputTargetsVector)
        J += self.__addRegularizationParameter(m, regParam)
        flatGrad = []
        grads = self.__backPropagation(layerOutputs, outputTargetsVector, m,
                                       regParam)
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

    def __minimize(self, costFunction, thetaInit, minimizationOptions, inputFeaturesVector, outputTargetsVector):
        # factory routine decides which optimizer to use
        # costFunction needs to return the objective value and the gradient in a list
        optimizer = minimizationOptions['optimizer']
        tol = minimizationOptions['tol']
        optimizationOptions = {'maxiter': minimizationOptions['maxiter']}
        if optimizer == 'CG':
            costFunctionCG = lambda theta : costFunction(theta, inputFeaturesVector, outputTargetsVector)
            optimizationOptions['learningRate'] = 1.0
            optimizationOptions['tol'] = tol
            res = self.__GD(costFunctionCG, thetaInit, optimizationOptions)
        elif optimizer == 'BSGD':
            optimizationOptions['learningRate'] = 1.0
            optimizationOptions['tol'] = tol
            res = self.__BSGD(costFunction, thetaInit, optimizationOptions, inputFeaturesVector, outputTargetsVector)
        else:
            optimizationOptions['gtol'] = tol
            costFunctionCG = lambda theta : costFunction(theta, inputFeaturesVector, outputTargetsVector)
            res = minimize(fun = costFunctionCG, x0 = thetaInit, method = 'L-BFGS-B', options = optimizationOptions, jac = True)
        return res

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

    def __BSGD(self, costFunction, thetaInit, optimizationOptions, inputFeaturesVector, outputTargetsVector):
        # This function performs a batch stochastic gradient descent
        learningRate = optimizationOptions['learningRate']
        maxiter = optimizationOptions['maxiter']
        tol = optimizationOptions['tol']
        vectorTheta = thetaInit

        res = {}
        res['success'] = False
        res['message'] = 'max iteration reached'

        l2GradDelta = 0.0
        m = inputFeaturesVector.__len__()
        batchSize = int(floor(m/2.0))

        for iterCounter in range(maxiter):
            batchIndex = sample(range(m), batchSize)
            [cost, grad] = costFunction(vectorTheta, inputFeaturesVector[batchIndex], outputTargetsVector[batchIndex])
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
