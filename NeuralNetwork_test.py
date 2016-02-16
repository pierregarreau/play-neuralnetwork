from numpy import array,sum,abs,zeros
from NeuralNetwork import NeuralNetwork
from NeuralNetworkUtil import NeuralNetworkUtil
from NeuralNetworkConfig import loadNNConfigA,loadNNConfigB, loadNNConfigC

inputVectors,targets,neuralNetworkArchitecture = loadNNConfigC()

def initialize_test():
    NeuralNetwork(neuralNetworkArchitecture)
    assert True == True

def backprop_test():
    # Set up of NN
    nn = NeuralNetwork(neuralNetworkArchitecture)
    nn._NeuralNetwork__initializeNeuralNetworkParameters()
    listOfThetas = nn.getListOfThetas()
    vectorTheta = NeuralNetworkUtil.roll(listOfThetas)
    # Test
    errTolerance = 0.0000001
    assert gradientCheck(nn, vectorTheta, inputVectors, targets) < errTolerance

### Beginning of helper functions

def gradientCheck(nn, vectorTheta, inputVectors, targets):

    numOfPointsTested = vectorTheta.__len__()
    def objectiveFunction(vectorTheta):
        return nn._NeuralNetwork__costFunctionOnly(vectorTheta, inputVectors, targets)

    # System under test
    [J, grad] = nn._NeuralNetwork__costFunction(vectorTheta, inputVectors, targets)
    numericalGradient = computeNumericalGradient(vectorTheta, objectiveFunction, numOfPointsTested)
    error = sum(abs(numericalGradient - grad[:numOfPointsTested]))
    print(grad, numericalGradient)
    print(error)
    return error

def computeNumericalGradient(vectorTheta, function, numOfPointsTested):
    epsilon = 0.0001
    pertubationVector = zeros(vectorTheta.__len__())
    Jup = zeros(numOfPointsTested)
    Jdown = zeros(numOfPointsTested)
    for index in range(numOfPointsTested):
        pertubationVector[index] = epsilon
        Jup[index] =  function(vectorTheta + pertubationVector)
        Jdown[index] = function(vectorTheta - pertubationVector)
        pertubationVector[index] = 0.0
    return 0.5 * (Jup - Jdown) / epsilon
