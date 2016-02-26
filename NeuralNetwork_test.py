from numpy import array,sum,abs,zeros,equal
from NeuralNetwork import NeuralNetwork
from NeuralNetworkUtil import NeuralNetworkUtil
from NeuralNetworkConfig import loadNNConfigA1, loadNNConfigB, loadNNConfigC

inputVectors,targets,neuralNetworkArchitecture = loadNNConfigC()

def initialize_test():
    NeuralNetwork(neuralNetworkArchitecture)
    assert True == True

def feedforward_test():
    # Set up of NN
    inputVectors_local,targets_local,neuralNetworkArchitecture_local = loadNNConfigB()
    nn = NeuralNetwork(neuralNetworkArchitecture_local)
    initializeThetasForConfigB(nn)
    print(nn.getListOfThetas())
    predictions = nn._NeuralNetwork__feedForward(inputVectors_local)
    assert NeuralNetworkUtil.computePredictionAccuracy(predictions[-1][1], targets_local) > 0.99

def predict_test():
    # Set up of NN
    inputVectors_local,targets_local,neuralNetworkArchitecture_local = loadNNConfigB()
    nn = NeuralNetwork(neuralNetworkArchitecture_local)
    initializeThetasForConfigB(nn)
    print(nn.getListOfThetas())
    predictions = nn.predict(inputVectors_local)
    assert NeuralNetworkUtil.computePredictionAccuracy(predictions, targets_local) > 0.99

def backpropGradient_test():
    # Set up of NN
    nn = NeuralNetwork(neuralNetworkArchitecture)
    nn._NeuralNetwork__initializeNeuralNetworkParameters()
    listOfThetas = nn.getListOfThetas()
    print(listOfThetas)
    vectorTheta = NeuralNetworkUtil.roll(listOfThetas)
    # Test
    errTolerance = 1e-6
    assert gradientCheck(nn, vectorTheta, inputVectors, targets) < errTolerance

# def backprop_test():
#     # Set up of NN
#     inputVectors_local,targets_local,neuralNetworkArchitecture_local = loadNNConfigB()
#     nn = NeuralNetwork(neuralNetworkArchitecture_local)
#     nn.train(inputVectors_local,targets_local)
#     predictions = nn.predict(inputVectors_local)
#     assert NeuralNetworkUtil.computePredictionAccuracy(predictions, targets_local) > 0.9


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
    [J, grad] = nn._NeuralNetwork__costFunction(vectorTheta, inputVectors, targets, 0.0)
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

def initializeThetasForConfigB(nn):
    nn._NeuralNetwork__thetas[0] = array([[-30,20,20],[10,-20,-20]])
    nn._NeuralNetwork__thetas[1] = array([[-10,20,20]])
    DEFAULT_DIRECTORY = './data/trainedData'
    nn._NeuralNetwork__exportTrainedParametersToFiles(DEFAULT_DIRECTORY)
