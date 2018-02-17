import numpy as np

from analytics.model import NeuralNet
from analytics.util import Loss
from analytics.optimizer import GradientDescent
from data.load import Load


def test_init():
    # nn arxi should be a list of dimentions or tuples
    arxitecture = [2, 2]
    nn = NeuralNet(arxitecture)
    assert True is True


def test_predict():
    # Set up of NN
    features, labels = Load.labelled_xnor()
    arxitecture = [2, 2, 1]
    nn = NeuralNet(arxitecture)
    # In Lieu of training
    nn.thetas[0] = np.array([[-30, 20, 20], [10, -20, -20]])
    nn.thetas[1] = np.array([[-10, 20, 20]])
    predictions = nn.predict(features)
    print(np.array(zip(predictions, labels)))
    assert Loss.accuracy(predictions, labels) > 0.99


def test_fit():
    features, labels = Load.labelled_xnor(sample_size=500)
    arxitecture = [2, 2, 1]
    nn = NeuralNet(arxitecture)
    options = {
        'learningRate': 1.0,
        'maxiter': 500,
        'tol': 1e-9,
        'jac': True
    }
    optimizer = GradientDescent(options)
    loss = Loss.crossentropy
    res = nn.fit(features, labels, optimizer, loss)
    print(res.__dict__())
    assert False


# def backpropGradientCheck_test():
#     # Set up of NN
#     nn = NeuralNet(NeuralNetArchitecture)
#     nn._NeuralNet__initializeNeuralNetParameters()
#     listOfThetas = nn.getListOfThetas()
#     print(listOfThetas)
#     vectorTheta = NeuralNetUtil.roll(listOfThetas)
#     # Test
#     errTolerance = 1e-6
#     assert gradientCheck(nn, vectorTheta, inputVectors, targets) < errTolerance

# def backprop_test():
#     # Set up of NN
#     features,labels,arxitecture = loadNNConfigB()
#     nn = NeuralNet(arxitecture)
#     nn.train(features,labels)
#     predictions = nn.predict(features)
#     assert NeuralNetUtil.computePredictionAccuracy(predictions, labels) > 0.9

# def gradientCheck(nn, vectorTheta, inputVectors, targets):
#     def objectiveFunction(vectorTheta):
#         return nn._NeuralNet__costFunctionOnly(vectorTheta, inputVectors,
#                                                    targets)

#     # System under test
#     [J, grad] = nn._NeuralNet__costFunction(vectorTheta, inputVectors,
#                                                 targets, 0.0)
#     numericalGradient = NeuralNetUtil.numerical_gradient(
#         vectorTheta, objectiveFunction)
#     error = np.sum(np.abs(numericalGradient - grad)) / grad.__len__()
#     print(np.array(zip(grad, numericalGradient)))
#     print(error)
#     return error
