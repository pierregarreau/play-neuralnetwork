from numpy import array,empty
from NeuralNetworkUtil import *
import random

DATA_DIRECTORY = './data'
DEFAULT_DIRECTORY = './data/trainedData'

def loadNNConfigA0():
    sampleSize = 100
    inputVectors = empty((sampleSize,2))
    targets = empty((sampleSize,1))
    for sample in range(sampleSize):
        inputVectors[sample] = [random.randint(0,1), random.randint(0,1)]
        targets[sample] = NotXandNotY(inputVectors[sample][0], inputVectors[sample][1])
    neuralNetworkArchitecture = array([2,1])
    return inputVectors,targets,neuralNetworkArchitecture

def loadNNConfigA1():
    sampleSize = 1000
    inputVectors = empty((sampleSize,2))
    targets = empty((sampleSize,1))
    for sample in range(sampleSize):
        inputVectors[sample] = [random.randint(0,1), random.randint(0,1)]
        targets[sample] = NotXandNotY(inputVectors[sample][0], inputVectors[sample][1])
    neuralNetworkArchitecture = array([2,1,1])
    return inputVectors,targets,neuralNetworkArchitecture

def loadNNConfigB():
    # XNOR Network
    sampleSize = 1000
    inputVectors = empty((sampleSize,2))
    targets = empty((sampleSize,1))
    for sample in range(sampleSize):
        inputVectors[sample] = [random.randint(0,1), random.randint(0,1)]
        targets[sample] = XNOR(inputVectors[sample][0], inputVectors[sample][1])
    neuralNetworkArchitecture = array([2,2,1])
    return inputVectors,targets,neuralNetworkArchitecture

def initializeThetasForConfigB(nn):
    nn._NeuralNetwork__thetas[0] = array([[-30,20,20],[10,-20,-20]])
    nn._NeuralNetwork__thetas[1] = array([[-10,20,20]])
    nn._NeuralNetwork__exportTrainedParametersToFiles(DEFAULT_DIRECTORY)

def loadNNConfigC():
    # XNOR Network
    sampleSize = 1000
    inputVectors = empty((sampleSize,2))
    targets = empty((sampleSize,2))
    for sample in range(sampleSize):
        inputVectors[sample] = [random.randint(0,1), random.randint(0,1)]
        targets[sample] = [XNOR(inputVectors[sample][0], inputVectors[sample][1]),NOR(inputVectors[sample][0], inputVectors[sample][1])]
    neuralNetworkArchitecture = array([2,2,2])
    return inputVectors,targets,neuralNetworkArchitecture

def loadNNConfigD0():
    inputVectorsFile = DATA_DIRECTORY + '/inputVectors.txt'
    inputVectors = NeuralNetworkUtil.loadDataFromFile(inputVectorsFile)

    targetsFile = DATA_DIRECTORY + '/targetVectors.txt'
    targets = NeuralNetworkUtil.loadDataFromFile(targetsFile)
    targets = NeuralNetworkUtil.transformClassificationTargetToBinaryVector(targets)

    dimension = 10
    neuralNetworkArchitecture = array([400,25,10])

    return inputVectors[:dimension],targets[:dimension],neuralNetworkArchitecture

def loadNNConfigD1():
    inputVectorsFile = DATA_DIRECTORY + '/inputVectors.txt'
    inputVectors = NeuralNetworkUtil.loadDataFromFile(inputVectorsFile)

    targetsFile = DATA_DIRECTORY + '/targetVectors.txt'
    targets = NeuralNetworkUtil.loadDataFromFile(targetsFile)
    targets = NeuralNetworkUtil.transformClassificationTargetToBinaryVector(targets)

    neuralNetworkArchitecture = array([400,25,10])

    return inputVectors,targets,neuralNetworkArchitecture

def OR(x,y):
    if ((x == 1)):
        return 1
    elif ((y == 1)):
        return 1
    else:
        return 0

def NotXandNotY(x,y):
    if ((x == 0) & (y == 0)):
        return 1
    else:
        return 0

def XNOR(x,y):
    if ((x == 1) & (y == 0)):
        return 0
    elif ((y == 1) & (x == 0)):
        return 0
    else:
        return 1

def NOR(x,y):
    if ((x == 1) & (y == 0)):
        return 1
    elif ((y == 1) & (x == 0)):
        return 1
    else:
        return 0
