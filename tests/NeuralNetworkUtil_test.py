from numpy import array,empty,equal
from NeuralNetworkUtil import NeuralNetworkUtil
from NeuralNetworkConfig import loadNNConfigB, loadNNConfigC

inputVectors,targets,neuralNetworkArchitecture = loadNNConfigC()

numEntries = 21
listThetas = [array([[0,1,2], [3,4,5], [6,7,8]]), \
              array([[9,10,11,12], [13,14,15,16], [17,18,19,20]])]
vectorTheta = array(range(numEntries))

def roll_test():
    inputFeaturesVector = NeuralNetworkUtil.roll(listThetas)
    print(inputFeaturesVector)
    assert inputFeaturesVector.__len__() == vectorTheta.__len__()
    for thetaTested, theta in zip(inputFeaturesVector,vectorTheta):
        assert thetaTested == theta

def unroll_test():
    emptyList = [empty((3,3)),empty((3,4))]
    NeuralNetworkUtil.unroll(vectorTheta, emptyList)
    for thetaTested, theta in zip(emptyList,listThetas):
        assert equal(thetaTested.shape[0], theta.shape[0])
        assert equal(thetaTested.shape[1], theta.shape[1])
        for row in range(theta.shape[0]):
            for col in range(theta.shape[1]):
                assert thetaTested[row,col] == theta[row,col]


def computePredictionAccuracy_test():
    targetValues = NeuralNetworkUtil.transformClassificationTargetToValue(targets)
    assert NeuralNetworkUtil.computePredictionAccuracy(targets,targetValues) == 1.0