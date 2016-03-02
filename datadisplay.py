import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from random import sample
from math import sqrt, floor
from NeuralNetworkConfig import *
from NeuralNetwork import NeuralNetwork
from NeuralNetwork_test import initializeThetasForConfigB

from numpy import array,append,arange,meshgrid

def plot(dataset, numImages):
    # This routines plots numImages images contained in the dataset
    squareSide = floor(sqrt(numImages))
    index = sample(range(dataset.__len__()), numImages)

    for i,plotnum in zip(index, range(numImages)):
        plt.subplot(squareSide, squareSide, plotnum)
        image = np.reshape(dataset[i],(20,20))
        imgplot = plt.imshow(image)
        imgplot.axes.get_xaxis().set_ticks([])
        imgplot.axes.get_yaxis().set_ticks([])
        imgplot.axes.get_xaxis().set_visible(False)
        imgplot.axes.get_yaxis().set_visible(False)
        imgplot.axes.get_xaxis().set_ticklabels([])
        imgplot.axes.get_yaxis().set_ticklabels([])

    plt.show(block=True)

### Plot function to study effect of regularization parameter on objectiveFunction

def plotCostFunction(regParam):
    # This function plots the cost function J as a function of
    # 2 parameters theta1 and theta2. This is for a given value of
    # the regularization parameter. For a high value of regParam, J should
    # converge to a paraboloid.
    inputVectors,targets,neuralNetworkArchitecture = loadNNConfigB()
    nn = NeuralNetwork(neuralNetworkArchitecture)
    X = arange(-15, 15.0, 0.5)
    Y = arange(-15, 15.0, 0.5)
    theta1, theta2 = meshgrid(X, Y)
    J = array([])
    for x,y in zip(theta1,theta2):
        for value in x:
            theta = array([-30.0, 20.0, 20.0, value, y[0], -20.0,-10,20.0,20.0])
            cost = nn._NeuralNetwork__costFunction(theta, inputVectors,targets, regParam)[0]
            J = append(J,cost)
    J = J.reshape(X.shape[0],X.shape[0])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(theta1, theta2, J)
    plt.show(block=True)

def plotCostFunctionVsRegParam():
    # This is a linear curve
    inputVectors,targets,neuralNetworkArchitecture = loadNNConfigB()
    nn = NeuralNetwork(neuralNetworkArchitecture)
    initializeThetasForConfigB(nn)
    vectorTheta = NeuralNetworkUtil.roll(nn.getListOfThetas())
    regParamSet = arange(0.1, 10.0, 0.4)
    trainResults = list()
    for regParam in regParamSet:
        trainResults.append(nn._NeuralNetwork__costFunction(vectorTheta, inputVectors,targets, regParam)[0])
    plt.plot(regParamSet, trainResults)
    plt.show()

def plotCostFunctionAfterTrainingVsRegParam():
    inputVectors,targets,neuralNetworkArchitecture = loadNNConfigA1()
    nn = NeuralNetwork(neuralNetworkArchitecture)
    regParamSet = arange(0.0, 10.0, 1.0)
    trainResults = list()
    for regParam in regParamSet:
        trainResults.append(nn.train(inputVectors,targets,regParam).fun)
    plt.plot(regParamSet, trainResults)
    plt.show()
