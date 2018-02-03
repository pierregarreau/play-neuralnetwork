from math import ceil, fabs
from numpy import loadtxt, savetxt, empty, ones, array, append, reshape, eye, exp, log, floor
import numpy as np


class NeuralNetworkUtil:
    def __init__():
        pass

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + exp(-z))

    @staticmethod
    def sigmoidGrad(z):
        return NeuralNetworkUtil.sigmoid(z) * (
            1.0 - NeuralNetworkUtil.sigmoid(z))

    @staticmethod
    def addBiasTerm(layerInput):
        # This function inserts a row of ones at position 0
        colSize = layerInput.shape[0]
        rowSize = layerInput.shape[1]
        layerInputWithBias = empty((colSize, rowSize + 1))
        layerInputWithBias[:, 0] = ones(colSize)
        layerInputWithBias[:, 1:] = layerInput
        return layerInputWithBias

    @staticmethod
    def applyScalarFunction(arrayInput, function):
        result = np.array(map(lambda value: function(value), arrayInput))
        # TODO this applyScalar may not be needed
        print('input shape: ', arrayInput.shape)
        print('result shape: ', result.shape)
        return result

    @staticmethod
    def computePredictionAccuracy(predictedTarget, targets):
        # target is (n,1) with values in 1, ..., m,
        # predictedTarget is (n,m) with values between 0 and 1
        epsilon = 1e-6
        targetValues = NeuralNetworkUtil.transformClassificationTargetToValue(
            targets)
        predictionValues = NeuralNetworkUtil.transformClassificationTargetToValue(
            predictedTarget)
        predictionAccuracy = 0.0
        for prediction, target in zip(predictionValues, targetValues):
            # print(prediction,target)
            if abs(prediction - target) < epsilon:
                predictionAccuracy += 1.0
        return predictionAccuracy / float(targets.__len__())

    @staticmethod
    def roll(listOfThetas):
        # this function transforms a list of matrices into a vector
        # TODO write test for roll unroll
        vectorThetas = array([])
        for theta in listOfThetas:
            currentLayerSize = theta.shape[0]
            nextLayerSize = theta.shape[1]
            vectorThetas = append(vectorThetas,
                                  theta.reshape(currentLayerSize,
                                                nextLayerSize))
        return vectorThetas

    @staticmethod
    def unroll(vectorThetas, listOfThetas):
        # this function transforms the vector vectorThetas into the list of matrices listOfThetas
        # TODO write test for roll unroll
        # Test should include making sure the output gets the correct input values
        pointerFlatTheta = 0
        numOfMatrices = listOfThetas.__len__()
        for index, theta in zip(range(numOfMatrices), listOfThetas):
            currentLayerSize = theta.shape[0]
            nextLayerSize = theta.shape[1]
            sizeFlatTheta = currentLayerSize * nextLayerSize
            vectorTheta = array(vectorThetas[pointerFlatTheta:
                                             pointerFlatTheta + sizeFlatTheta])
            listOfThetas[index] = vectorTheta.reshape(currentLayerSize,
                                                      nextLayerSize)
            pointerFlatTheta += sizeFlatTheta

    @staticmethod
    def logErrorClassificationFunction(predictions, targets):
        assert predictions.__len__() == targets.__len__()
        J = 0.0
        m = targets.__len__()
        for target, prediction in zip(targets, predictions):
            J -= np.sum(target * log(prediction) + (1-target) * log(1 - prediction) )
        return J / m

    @staticmethod
    def transformClassificationTargetToValue(targets):
        if targets.shape.__len__() > 1:
            if targets.shape[1] > 1:
                return np.array(list(map(lambda x: x.argmax() + 1, targets)))
            else:
                return np.array(list(map(lambda x: np.floor(x + 0.5), targets)))
        else:
            return np.array(list(map(lambda x: np.floor(x + 0.5), targets)))

    @staticmethod
    def computeNumericalGradient(vectorTheta, function):
        epsilon = 1e-4
        numOfPoints = vectorTheta.__len__()
        pertubationVector = np.zeros(numOfPoints)
        Jup = np.zeros(numOfPoints)
        Jdown = np.zeros(numOfPoints)
        for index in range(numOfPoints):
            pertubationVector[index] = epsilon
            Jup[index] = function(vectorTheta + pertubationVector)
            Jdown[index] = function(vectorTheta - pertubationVector)
            pertubationVector[index] = 0.0
        return 0.5 * (Jup - Jdown) / epsilon
