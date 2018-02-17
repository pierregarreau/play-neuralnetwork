import numpy as np


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
        targetValues = NeuralNetworkUtil.transformClassificationTargetToValue(
            targets)
        predictionValues = NeuralNetworkUtil.transformClassificationTargetToValue(
            predictions)
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

    # deprecated below
    # ////////////////
    @staticmethod
    def applyScalarFunction(arrayInput, function):
        result = np.array(map(lambda value: function(value), arrayInput))
        # TODO this applyScalar may not be needed
        print('input shape: ', arrayInput.shape)
        print('result shape: ', result.shape)
        return result

    @staticmethod
    def roll(listOfThetas):
        # this function transforms a list of matrices into a vector
        # TODO write test for roll unroll
        vectorThetas = np.array([])
        for theta in listOfThetas:
            currentLayerSize = theta.shape[0]
            nextLayerSize = theta.shape[1]
            vectorThetas = np.append(vectorThetas,
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
            vectorTheta = np.array(vectorThetas[pointerFlatTheta:
                                             pointerFlatTheta + sizeFlatTheta])
            listOfThetas[index] = vectorTheta.reshape(currentLayerSize,
                                                      nextLayerSize)
            pointerFlatTheta += sizeFlatTheta
