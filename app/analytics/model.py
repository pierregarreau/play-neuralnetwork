import numpy as np
import math as math

from abc import ABCMeta, abstractclassmethod
from typing import Dict, List, Callable, Tuple

from analytics.optimizer import Optimizer
from analytics.util import Activation, NeuralNetworkUtil
from data.load import Load

DEFAULT_DIRECTORY = './data/trainedData'

# TODO NeuralNetwork is a factory?


class Model(metaclass=ABCMeta):

    @abstractclassmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        pass

    @abstractclassmethod
    def fit(self, features: np.ndarray, labels: np.ndarray, optimizer: Optimizer, loss: Callable[[np.ndarray], float]) -> Dict:
        pass

    def save(self, file_name: str) -> None:
        pass

    def load(self, file_name: str) -> None:
        pass


class NeuralNet(Model):
    def __init__(self, layers: List):
        if len(layers) == 0:
            print('Error: need at least one layer to build this network')
        else:
            self.layers = layers
            self.n_layers = len(layers)
            self.n_parameters = sum([next_layer * (layer + 1) for layer, next_layer in zip(layers[:-1], layers[1:])])
            self.theta = np.empty(self.n_parameters)
            self.thetas = []
            self._decompose(self.theta, self.thetas)

    def predict(self, features: np.ndarray) -> np.ndarray:
        targetPrediction = self._feed_forward(features)
        return targetPrediction[-1][1]

    def fit(self, features: np.ndarray, labels: np.ndarray, optim: Optimizer, loss: Callable[[np.ndarray], float]) -> Dict:
        self._random_init()

        def objective(theta):
            return self.objective(theta, features, labels, loss)

        res = optim.minimize(objective=objective, init=self.theta)
        return res

    def _decompose(self, theta: np.ndarray, thetas: List) -> List[np.ndarray]:
        idx = 0
        for layer, next_layer in zip(self.layers[:-1], self.layers[1:]):
            n_params = next_layer * (layer + 1)
            thetas.append(theta[idx:idx+n_params].reshape((next_layer, layer + 1)))
            idx += n_params

    def _random_init(self) -> None:
        for index, theta in zip(range(self.n_layers), self.thetas):
            current_layer_size = theta.shape[0]
            next_layer_size = theta.shape[1]
            epsilon = math.sqrt(6.0) / math.sqrt(current_layer_size + next_layer_size)
            self.thetas[index] = epsilon * (np.random.uniform(0, 2.0, (current_layer_size, next_layer_size)) - 1.0)

    def _feed_forward(self, features: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        a = NeuralNetworkUtil.add_bias(features)
        predicted = list()
        predicted.append(([], a))
        for layerIndex, theta in zip(range(self.n_layers), self.thetas):
            z = np.dot(a, theta.transpose())
            # TODO refactor this sigmoid into function
            sigmoid = Activation.sigmoid(-z)
            if layerIndex < self.n_layers - 2:
                a = NeuralNetworkUtil.add_bias(sigmoid)
            else:
                a = sigmoid
            predicted.append((z, a))
        return predicted

    def objective(self, theta, features, labels, loss: Callable[[np.ndarray], float], omega: float = 0.1) -> List[np.ndarray]:
        J = []
        dJ = []
        m = features.__len__()
        self.theta = theta
        self.thetas = []
        self._decompose(theta, self.thetas)

        # predict + compute loss + regularization = objective
        predicted = self._feed_forward(features)
        J = loss(predicted[-1][1], labels)
        self.regularize(J, m, omega)

        # backprop + regularization = Dobjective
        dJ, dJs = self._back_propagate(predicted, labels, m, omega)
        self.regularize_gradient(dJs, m, omega)
        # flatGrad = roll(dJ)

        return [J, dJ]

    def _back_propagate(self, predicted: List[Tuple[np.ndarray]], labels: np.ndarray) -> [np.ndarray, List[np.ndarray]]:
        # Initialization
        m = labels.shape[0]
        grad = np.empty(self.n_parameters)
        grads = []
        self._decompose(grad, grads)

        # First iteration
        delta = predicted[-1][1] - labels
        biased_a2 = predicted[-2][1]
        grads[-1][:, :] = np.dot(delta.transpose(), biased_a2) / m

        # Other layers
        for index, theta in reversed(list(enumerate(self.thetas))):
            if index > 0:
                propError = np.dot(delta, theta)
                gZ = Activation.dsigmoid(predicted[index][0])
                delta = np.multiply(gZ, propError[:, 1:])
                grads[index-1][:, :] = np.dot(delta.transpose(), predicted[index-1][1]) / m

        return grad, grads

    def regularize(self, J: float, m: int, omega) -> float:
        Jreg = 0.0
        for theta in self.thetas:
            Jreg += np.sum(theta[:, 1:] * theta[:, 1:])
        J += 0.5 * omega * Jreg / m

    def regularize_gradient(self, dJs: np.ndarray, m, omega):
        # for index, theta in zip(range(self.n_layers), self.thetas):
        assert len(dJs) == len(self.thetas)
        for dJ, theta in zip(dJs, self.thetas):
        # for idx in range(self.n_layers-1):
            # assert dJs[idx].shape == self.thetas[idx].shape
            assert dJ.shape == theta.shape
            dJ[:, 1:] += omega * theta[:, 1:] / m

    def __minimize(self, costFunction, thetaInit, minimizationOptions, features, labels):
        # factory routine decides which optimizer to use
        # costFunction needs to return the objective value and the gradient in a list
        #
        # This function has been retired
        #
        optimizer = minimizationOptions['optimizer']
        tol = minimizationOptions['tol']
        optimizationOptions = {'maxiter': minimizationOptions['maxiter']}
        if optimizer == 'CG':
            costFunctionCG = lambda theta : costFunction(theta, features, labels)
            optimizationOptions['learningRate'] = 1.0
            optimizationOptions['tol'] = tol
            res = self.__GD(costFunctionCG, thetaInit, optimizationOptions)
        elif optimizer == 'BSGD':
            optimizationOptions['learningRate'] = 1.0
            optimizationOptions['tol'] = tol
            res = self.__BSGD(costFunction, thetaInit, optimizationOptions, features, labels)
        else:
            optimizationOptions['gtol'] = tol
            costFunctionCG = lambda theta : costFunction(theta, features, labels)
            res = minimize(fun = costFunctionCG, x0 = thetaInit, method = 'L-BFGS-B', options = optimizationOptions, jac = True)
        return res

    # Save / load
    #
    #
    def __loadTrainedParametersFromFiles(self, trainedParametersDirectory):
        # TODO check if the dimensions of the loaded parameters are the ones announced
        # in the construction of the neural network
        for counter in range(self.n_layers - 1):
            trainedThetaFileName = trainedParametersDirectory + '/Theta' + str(
                counter) + '.txt'
            theta = Load.from_file(trainedThetaFileName)
            if theta.shape.__len__() > 1:
                self.thetas[counter] = theta
            else:
                self.thetas[counter] = theta.reshape(1, theta.size)

    def __exportTrainedParametersToFiles(self, trainedParametersDirectory):
        # TODO check if the dimensions of the loaded parameters are the ones announced
        # in the construction of the neural network
        for counter, theta in zip(range(self.n_layers - 1), self.thetas):
            trainedThetaFileName = trainedParametersDirectory + '/Theta' + str(
                counter) + '.txt'
            Load.to_file(theta, trainedThetaFileName)
    #
    #

    # Below only used for computing numerial gradient (testing)
    def _feed_forwardNoHistory(self, features):
        a = features
        for theta in self.thetas:
            layerInputWithBias = NeuralNetworkUtil.add_bias(a)
            z = np.dot(layerInputWithBias,theta.transpose())
            a = NeuralNetworkUtil.applyScalarFunction(z, NeuralNetworkUtil.sigmoid)
        return (z,a)

    def objectiveOnly(self, theta, features, labels):
        J = []
        unroll(theta, self.thetas)
        predicted = self._feed_forwardNoHistory(features)
        J = NeuralNetworkUtil.logErrorClassificationFunction(predicted[1], labels)
        return J
