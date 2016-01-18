import numpy as np

DEFAULT_SAVE_FILE = './data/trainedParameters.txt'

class NeuralNetwork:

    def __init__(self, numOfNodesPerLayer):
        # numOfNodesPerLayer is an array of integers giving the number of nodes per layer
        self.__construct(numOfNodesPerLayer)

    def predict(self, inputFeaturesVector, trainedParametersFile = DEFAULT_SAVE_FILE):
        # this function performs forward propagation
        pass

    def train(self, inputFeaturesVector, outputTargetsVector, trainedParametersFile = DEFAULT_SAVE_FILE):
        # this function trains the neural network with backward propagation
        pass

    def __construct(self, numOfNodesPerLayer):
        if numOfNodesPerLayer.ndim > 1:
            print('Error : numOfNodesPerLayer needs to be a vector')
        else:
            self.numLayers = numOfNodesPerLayer.size
            self.theta = [np.empty((currentLayer, nextLayer))
                for currentLayer, nextLayer in zip(numOfNodesPerLayer[:-1], numOfNodesPerLayer[1:])]
