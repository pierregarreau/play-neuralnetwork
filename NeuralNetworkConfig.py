from numpy import array,empty
from NeuralNetworkUtil import *
import random

DATA_DIRECTORY = './data'

def loadNNConfigA():
    inputVectors = array([[0, 0],[1, 0],[0, 1],[1, 1],[0, 0],[1, 0],[0, 1],[1, 1],[0, 0],[1, 0], [0, 1], [1, 1]])
    targets = array([[0],[0],[0],[1],[0],[0],[0],[1],[0],[0],[0],[1]])
    neuralNetworkArchitecture = array([2,2,1])
    return inputVectors,targets,neuralNetworkArchitecture

def loadNNConfigB():
    # XNOR Network
    sampleSize = 5000
    inputVectors = empty((sampleSize,2))
    targets = empty((sampleSize,1))
    for sample in range(sampleSize):
        inputVectors[sample] = [random.randint(0,1), random.randint(0,1)]
        targets[sample] = XNOR(inputVectors[sample][0], inputVectors[sample][1])
    neuralNetworkArchitecture = array([2,6,1])
    return inputVectors,targets,neuralNetworkArchitecture

def loadNNConfigC():
    # XNOR Network
    sampleSize = 5000
    inputVectors = empty((sampleSize,2))
    targets = empty((sampleSize,2))
    for sample in range(sampleSize):
        inputVectors[sample] = [random.randint(0,1), random.randint(0,1)]
        targets[sample] = [XNOR(inputVectors[sample][0], inputVectors[sample][1]),NOR(inputVectors[sample][0], inputVectors[sample][1])]
    neuralNetworkArchitecture = array([2,6,2])
    return inputVectors,targets,neuralNetworkArchitecture

def loadNNConfigD():
    inputVectorsFile = DATA_DIRECTORY + '/inputVectors.txt'
    inputVectors = NeuralNetworkUtil.loadDataFromFile(inputVectorsFile)

    targetsFile = DATA_DIRECTORY + '/targetVectors.txt'
    targets = NeuralNetworkUtil.loadDataFromFile(targetsFile)
    targets = NeuralNetworkUtil.transformClassificationTargetToBinaryVector(targets)

    neuralNetworkArchitecture = array([400,25,10])

    return inputVectors,targets,neuralNetworkArchitecture

def XNOR(x,y):
    if ((x == 1) & (y == 0)):
        return 1
    elif ((y == 1) & (x == 0)):
        return 1
    else:
        return 0

def NOR(x,y):
    if ((x == 1) & (y == 0)):
        return 0
    elif ((y == 1) & (x == 0)):
        return 0
    else:
        return 1
