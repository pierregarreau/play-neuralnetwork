#! /usr/bin/env python

import numpy as np

from sklearn.model_selection import train_test_split
from analytics.model import NeuralNet
from analytics.optimizer import GradientDescent, Optimizer
from analytics.util import Loss
from data.load import Load

if __name__ == "__main__":
    # Data
    features, labels = Load.labelled_xnor(100)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33)
    # Fit
    layers = [2, 2, 1]
    neural_net = NeuralNet(layers)
    optimizer = GradientDescent(options={
        'optimizer': '',
        'maxiter': 1000,
        'tol': 1e-7,
        'jac': True,
        'learning_rate': 1.0
    })
    loss = Loss.crossentropy
    res = neural_net.fit(X_train, y_train, optimizer, loss)
    print(res)
    # Predict
    predicted = neural_net.predict(X_test)
    for p, y in zip(predicted, y_test):
        print(p,y)
    loss = loss(predicted, y_test)
    print('Loss: ', loss)

