import numpy as np
import random as rnd

from data.util import integer_class_to_binary_vector, not_x_and_not_y, xnor, nor

DATA_DIRECTORY = 'data/'
DEFAULT_DIRECTORY = 'data/trainedData/'


class Load:
    @staticmethod
    def labelled_not_and_not(sample_size: int = 100) -> [np.array, np.array]:
        features = np.empty((sample_size, 2))
        labels = np.empty((sample_size, 1))
        for sample in range(sample_size):
            features[sample] = [rnd.randint(0, 1), rnd.randint(0, 1)]
            labels[sample] = not_x_and_not_y(features[sample][0],
                                             features[sample][1])
        # A0: neuralNetworkArchitecture = np.array([2,1])
        # A1: neuralNetworkArchitecture = np.array([2,1,1])
        return features, labels

    @staticmethod
    def labelled_xnor(sample_size: int = 100) -> [np.array, np.array]:
        features = np.empty((sample_size, 2))
        labels = np.empty((sample_size, 1))
        for sample in range(sample_size):
            features[sample] = [rnd.randint(0, 1), rnd.randint(0, 1)]
            labels[sample] = xnor(features[sample][0], features[sample][1])
        # neuralNetworkArchitecture = np.array([2,2,1])
        return features, labels

    @staticmethod
    def labelled_xnor_nor(sample_size: int = 100) -> [np.array, np.array]:
        features = np.empty((sample_size, 2))
        labels = np.empty((sample_size, 2))
        for sample in range(sample_size):
            features[sample] = [rnd.randint(0, 1), rnd.randint(0, 1)]
            labels[sample] = [
                xnor(features[sample][0], features[sample][1]),
                nor(features[sample][0], features[sample][1])
            ]
        # neuralNetworkArchitecture = np.array([2,2,2])
        return np.array(features), np.array(labels)

    @staticmethod
    def mnist(sample_size: int = None) -> [np.array, np.array]:
        # neuralNetworkArchitecture = np.array([400,25,10])
        features_file = DATA_DIRECTORY + 'features.txt'
        labels_file = DATA_DIRECTORY + 'targetVectors.txt'
        features = Load.from_file(features_file)
        labels = Load.from_file(labels_file)
        labels = integer_class_to_binary_vector(labels)
        if sample_size:
            return features[:sample_size], labels[:sample_size]
        return features, labels

    @staticmethod
    def from_file(file_path):
        # This function reads a file and returns a numpy array
        return np.loadtxt(file_path)

    @staticmethod
    def to_file(theta, file_path):
        # This function reads a file and returns a numpy array
        return np.savetxt(file_path, theta)
