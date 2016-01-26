
from math import exp,ceil
from numpy import loadtxt, empty, ones, array, append, reshape

class NeuralNetworkUtil:
    def __init__():
        pass

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + exp(-z))

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
    def applyScalarFunction(array, function):
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                array[i,j] = function(array[i,j])

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
        vectorThetas = array([])
        for theta in listOfThetas:
            currentLayerSize = theta.shape[0]
            nextLayerSize = theta.shape[1]
            vectorThetas = append(vectorThetas, theta.reshape(currentLayerSize, nextLayerSize))
            print("vector theta",theta.reshape(currentLayerSize, nextLayerSize))
        return vectorThetas

    @staticmethod
    def unroll(vectorThetas, listOfThetas):
        # this function transforms a vector into a list of matrices
        pointerFlatTheta = 0
        for theta in listOfThetas:
            currentLayerSize = theta.shape[0]
            nextLayerSize = theta.shape[1]
            sizeFlatTheta = currentLayerSize * nextLayerSize
            vectorTheta = vectorThetas[pointerFlatTheta:pointerFlatTheta+sizeFlatTheta]
            theta = vectorTheta.reshape(nextLayerSize,currentLayerSize)
            pointerFlatTheta += sizeFlatTheta

    @staticmethod
    def leastSquareError(prediction, target):
        # TODO
        return 1.0
