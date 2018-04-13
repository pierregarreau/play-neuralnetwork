import numpy as np

from data.util import binary_vector_to_class


class Activation:
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def dsigmoid(z):
        return Activation.sigmoid(z) * (1.0 - Activation.sigmoid(z))


class Loss:
    @staticmethod
    def crossentropy(predictions: np.ndarray, targets: np.ndarray) -> float:
        assert predictions.__len__() == targets.__len__()
        J = 0.0
        m = targets.__len__()
        for target, prediction in zip(targets, predictions):
            J -= np.sum(target * np.log(prediction) + (1-target) * np.log(1 - prediction))
        return J / m

    @staticmethod
    def squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
        assert predictions.__len__() == targets.__len__()
        m = targets.__len__()
        diff = (predictions - targets)
        return sum(diff[:] * diff[:]) / 2.0 / m

    @staticmethod
    def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
        # target is (n,1) with values in 1, ..., m,
        # predictions is (n,m) with values between 0 and 1
        epsilon = 1e-6
        targetValues = binary_vector_to_class(targets)
        predictionValues = binary_vector_to_class(predictions)
        predictionAccuracy = 0.0
        for prediction, target in zip(predictionValues, targetValues):
            # print(prediction,target)
            if abs(prediction - target) < epsilon:
                predictionAccuracy += 1.0
        return predictionAccuracy / float(targets.__len__())


class NeuralNetworkUtil:
    def __init__():
        pass

    @staticmethod
    def add_bias(features: np.ndarray) -> np.array:
        # This function inserts a row of ones at position 0
        cols = features.shape[0]
        rows = features.shape[1]
        features_w_bias = np.empty((cols, rows + 1))
        features_w_bias[:, 0] = np.ones(cols)
        features_w_bias[:, 1:] = features
        return features_w_bias
