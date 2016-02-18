from numpy import array,sum,abs,zeros,equal
from NeuralNetwork import NeuralNetwork
from NeuralNetworkUtil import NeuralNetworkUtil
from NeuralNetworkConfig import loadNNConfigA,loadNNConfigB, loadNNConfigC, loadNNConfigD

inputVectors,targets,neuralNetworkArchitecture = loadNNConfigD()

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
    errTolerance = 1e-6
    assert gradientCheck(nn, vectorTheta, inputVectors, targets) < errTolerance

def numericalGradient_test():
    def linear(x):
        return sum(x)
    def square(x):
        return sum(x*x) / 2.0
    def cube(x):
        return sum(x*x*x) / 3.0
    testArray = array([-1.0, 0.0, 1.0, 2.0, 10.0])
    tolerance = 1e-8
    for sut, expected in zip(computeNumericalGradient(testArray, linear), array([1.0, 1.0, 1.0, 1.0, 1.0])):
        print(sut, expected)
        assert abs(sut-expected) < tolerance
    for sut, expected in zip(computeNumericalGradient(testArray, square), testArray):
        assert abs(sut-expected) < tolerance
    for sut, expected in zip(computeNumericalGradient(testArray, cube), array([1.0, 0.0, 1.0, 4.0, 100.0])):
        assert abs(sut-expected) < tolerance

### Beginning of helper functions

def gradientCheck(nn, vectorTheta, inputVectors, targets):
    def objectiveFunction(vectorTheta):
        return nn._NeuralNetwork__costFunctionOnly(vectorTheta, inputVectors, targets)

    # System under test
    [J, grad] = nn._NeuralNetwork__costFunction(vectorTheta, inputVectors, targets)
    numericalGradient = computeNumericalGradient(vectorTheta, objectiveFunction)
    error = sum(abs(numericalGradient - grad)) / grad.__len__()
    print(array(zip(grad, numericalGradient)))
    print(error)
    return error

def computeNumericalGradient(vectorTheta, function):
    epsilon = 1e-4
    numOfPoints = vectorTheta.__len__()
    pertubationVector = zeros(numOfPoints)
    Jup = zeros(numOfPoints)
    Jdown = zeros(numOfPoints)
    for index in range(numOfPoints):
        pertubationVector[index] = epsilon
        Jup[index] =  function(vectorTheta + pertubationVector)
        Jdown[index] = function(vectorTheta - pertubationVector)
        pertubationVector[index] = 0.0
    return 0.5 * (Jup - Jdown) / epsilon
