
from math import ceil,fabs
from numpy import loadtxt, empty, ones, array, append, reshape, eye, exp, log
import numpy as np

class NeuralNetworkUtil:
    def __init__():
        pass

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + exp(-z))

    @staticmethod
    def sigmoidGrad(z):
        return NeuralNetworkUtil.sigmoid(z) * (1.0 - NeuralNetworkUtil.sigmoid(z))

    @staticmethod
    def addBiasTerm(layerInput):
        # This function inserts a row of ones at position 0
        colSize = layerInput.shape[0]
        rowSize = layerInput.shape[1]
        layerInputWithBias = empty((colSize,rowSize+1))
        layerInputWithBias[:,0] = ones(colSize)
        layerInputWithBias[:,1:] = layerInput
        return layerInputWithBias

    @staticmethod
    def loadDataFromFile(pathToFile):
        # This function reads a file and returns a numpy array
        return loadtxt(pathToFile)

    @staticmethod
    def applyScalarFunction(arrayInput, function):
        return array(map(lambda value: function(value), arrayInput))

    @staticmethod
    def computePredictionAccuracy(predictedTarget,targets):
        # target is (n,1) with values in 1, ..., m,
        # predictedTarget is (n,m) with values between 0 and 1
        predictionAccuracy = 0.0
        n = predictedTarget.shape[0]
        m = predictedTarget.shape[1]
        for counter,target in zip(range(n),targets):
            if target%(predictedTarget[counter].argmax()+1)==0:
                predictionAccuracy += 1.0
        return predictionAccuracy / n

    @staticmethod
    def roll(listOfThetas):
        # this function transforms a list of matrices into a vector
        # TODO write test for roll unroll
        vectorThetas = array([])
        for theta in listOfThetas:
            currentLayerSize = theta.shape[0]
            nextLayerSize = theta.shape[1]
            vectorThetas = append(vectorThetas, theta.reshape(currentLayerSize, nextLayerSize))
        return vectorThetas

    @staticmethod
    def unroll(vectorThetas, listOfThetas):
        # this function transforms the vector vectorThetas into the list of matrices listOfThetas
        # TODO write test for roll unroll
        # Test should include making sure the output gets the correct input values
        pointerFlatTheta = 0
        numOfMatrices = listOfThetas.__len__()
        for index, theta in zip(range(numOfMatrices),listOfThetas):
            currentLayerSize = theta.shape[0]
            nextLayerSize = theta.shape[1]
            sizeFlatTheta = currentLayerSize * nextLayerSize
            vectorTheta = vectorThetas[pointerFlatTheta:pointerFlatTheta+sizeFlatTheta]
            listOfThetas[index] = vectorTheta.reshape(currentLayerSize, nextLayerSize)
            pointerFlatTheta += sizeFlatTheta

    @staticmethod
    def logErrorClassificationFunction(prediction, target):
        assert prediction.__len__() == target.__len__()
        return np.sum(np.sum( - target * log(prediction) - (1-target) * log(1 - prediction)) / target[0].__len__() ) / target.__len__()

    @staticmethod
    def transformClassificationTargetToBinaryVector(targets):
        maxTarget = targets.max()
        identityMatrix = eye(maxTarget,maxTarget)
        return array(map(lambda x : identityMatrix[x-1, :], targets))

    @staticmethod
    def transformClassificationTargetToValue(targets):
        return map(lambda x : x.max(), targets)
