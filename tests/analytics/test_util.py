import numpy as np

from analytics.NeuralNetworkUtil import NeuralNetworkUtil
from data.load import Load

inputVectors, targets = Load.labelled_xnor_nor()
neuralNetworkArchitecture = np.array([2, 2, 2])

numEntries = 21
listThetas = [np.array([[0,1,2], [3,4,5], [6,7,8]]), \
              np.array([[9,10,11,12], [13,14,15,16], [17,18,19,20]])]
vectorTheta = np.array(range(numEntries))


def numericalGradient_test():
    def linear(x):
        return np.sum(x)

    def square(x):
        return np.sum(x * x) / 2.0

    def cube(x):
        return np.sum(x * x * x) / 3.0

    x = np.array([-1.0, 0.0, 1.0, 2.0, 10.0])
    DlinearDx = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    DsquareDx = np.array([-1.0, 0.0, 1.0, 2.0, 10.0])
    DcubeDx = np.array([1.0, 0.0, 1.0, 4.0, 100.0])

    tolerance = 1e-8
    for sut, expected in zip(
            NeuralNetworkUtil.computeNumericalGradient(x, linear), DlinearDx):
        assert np.abs(sut - expected) < tolerance
    for sut, expected in zip(
            NeuralNetworkUtil.computeNumericalGradient(x, square), DsquareDx):
        assert np.abs(sut - expected) < tolerance
    for sut, expected in zip(
            NeuralNetworkUtil.computeNumericalGradient(x, cube), DcubeDx):
        assert np.abs(sut - expected) < tolerance


def roll_test():
    inputFeaturesVector = NeuralNetworkUtil.roll(listThetas)
    print(inputFeaturesVector)
    assert inputFeaturesVector.__len__() == vectorTheta.__len__()
    for thetaTested, theta in zip(inputFeaturesVector, vectorTheta):
        assert thetaTested == theta


def unroll_test():
    emptyList = [np.empty((3, 3)), np.empty((3, 4))]
    NeuralNetworkUtil.unroll(vectorTheta, emptyList)
    for thetaTested, theta in zip(emptyList, listThetas):
        assert np.equal(thetaTested.shape[0], theta.shape[0])
        assert np.equal(thetaTested.shape[1], theta.shape[1])
        for row in range(theta.shape[0]):
            for col in range(theta.shape[1]):
                assert thetaTested[row, col] == theta[row, col]


def computePredictionAccuracy_test():
    targetValues = NeuralNetworkUtil.transformClassificationTargetToValue(
        targets)
    assert NeuralNetworkUtil.computePredictionAccuracy(targets,
                                                       targetValues) == 1.0
