import numpy as np

from analytics.NeuralNetwork import NeuralNetwork
from analytics.NeuralNetworkUtil import NeuralNetworkUtil
from data.load import Load


def initialize_test():
    # nn arxi should be a list of dimentions or tuples
    NeuralNetwork(np.array([2, 2]))
    assert True is True


def feedforward_test():
    # Set up of NN
    features, labels = Load.labelled_xnor()
    nn_dimensions = np.array([2, 2, 1])
    nn = NeuralNetwork(nn_dimensions)
    # In Lieu of training
    nn._NeuralNetwork__thetas[0] = np.array([[-30, 20, 20], [10, -20, -20]])
    nn._NeuralNetwork__thetas[1] = np.array([[-10, 20, 20]])
    predictions = nn._NeuralNetwork__feedForward(features)
    print('Thetas', nn.getListOfThetas())
    print(np.array(zip(predictions[-1][1], labels)))
    assert NeuralNetworkUtil.computePredictionAccuracy(predictions[-1][1],
                                                       labels) > 0.99


def predict_test():
    # Set up of NN
    features, labels = Load.labelled_xnor()
    nn_dimensions = np.array([2, 2, 1])
    nn = NeuralNetwork(nn_dimensions)
    # In Lieu of training
    nn._NeuralNetwork__thetas[0] = np.array([[-30, 20, 20], [10, -20, -20]])
    nn._NeuralNetwork__thetas[1] = np.array([[-10, 20, 20]])
    predictions = nn.predict(features)
    assert NeuralNetworkUtil.computePredictionAccuracy(predictions,
                                                       labels) > 0.99


# def backpropGradientCheck_test():
#     # Set up of NN
#     nn = NeuralNetwork(neuralNetworkArchitecture)
#     nn._NeuralNetwork__initializeNeuralNetworkParameters()
#     listOfThetas = nn.getListOfThetas()
#     print(listOfThetas)
#     vectorTheta = NeuralNetworkUtil.roll(listOfThetas)
#     # Test
#     errTolerance = 1e-6
#     assert gradientCheck(nn, vectorTheta, inputVectors, targets) < errTolerance

# def backprop_test():
#     # Set up of NN
#     features,labels,nn_dimensions = loadNNConfigB()
#     nn = NeuralNetwork(nn_dimensions)
#     nn.train(features,labels)
#     predictions = nn.predict(features)
#     assert NeuralNetworkUtil.computePredictionAccuracy(predictions, labels) > 0.9

# def gradientCheck(nn, vectorTheta, inputVectors, targets):
#     def objectiveFunction(vectorTheta):
#         return nn._NeuralNetwork__costFunctionOnly(vectorTheta, inputVectors,
#                                                    targets)

#     # System under test
#     [J, grad] = nn._NeuralNetwork__costFunction(vectorTheta, inputVectors,
#                                                 targets, 0.0)
#     numericalGradient = NeuralNetworkUtil.computeNumericalGradient(
#         vectorTheta, objectiveFunction)
#     error = np.sum(np.abs(numericalGradient - grad)) / grad.__len__()
#     print(np.array(zip(grad, numericalGradient)))
#     print(error)
#     return error
