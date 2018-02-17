import numpy as np

from sklearn.model_selection import train_test_split

from analytics.model import NeuralNet
from analytics.optimizer import GradientDescent, Optimizer
from analytics.util import Loss
from data.load import Load


def test_init():
    # nn arxi should be a list of dimentions or tuples
    arxitecture = [2, 2]
    nn = NeuralNet(arxitecture)
    assert True is True


def test_predict():
    # Set up of NN
    features, labels = Load.labelled_xnor()
    arxitecture = [2, 2, 1]
    nn = NeuralNet(arxitecture)
    # In Lieu of training
    nn.theta = np.array([[-30, 20, 20], [10, -20, -20], [-10, 20, 20]])
    nn.thetas = nn._decompose(nn.theta)
    # Predict
    predictions = nn.predict(features)
    print(np.array(zip(predictions, labels)))
    assert Loss.accuracy(predictions, labels) > 0.99


def test_back_propagate():
    # Set-up
    layers = [2, 2, 1]
    neural_net = NeuralNet(layers)
    neural_net.theta = np.array([-30, 20, 20, 10, -20, -20, -10, 20, 20])
    neural_net.thetas = neural_net._decompose(neural_net.theta)
    loss = Loss.crossentropy
    # Backprop
    m = 100
    features, labels = Load.labelled_xnor(sample_size = m)
    predictions = neural_net._feed_forward(features)
    dJ, dJs = neural_net._back_propagate(predictions, labels, m)

    # numerical_gradient
    def objective(theta):
        layers = [2, 2, 1]
        nn = NeuralNet(layers)
        nn.theta = theta
        nn.thetas = nn._decompose(theta)
        predicted = nn._feed_forward(features)
        return loss(predicted[-1][1], labels)

    theta = np.array([-30, 20, 20, 10, -20, -20, -10, 20, 20])
    num_dJ = Optimizer.numerical_gradient(theta, objective)

    assert abs(np.sum(num_dJ - dJ)) < 1e-4


def test_fit():
    # DATA_DIRECTORY
    features, labels = Load.labelled_xnor()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33)

    print(X_train[:2], y_train[:2])

    # Fit
    layers = [2, 1]
    neural_net = NeuralNet(layers)
    optimizer = GradientDescent(options={'optimizer': '', 'maxiter': 1000, 'tol': 1e-7, 'jac': True})
    loss = Loss.crossentropy
    res = neural_net.fit(X_train, y_train, optimizer, loss)
    print(res.__dict__())
    # Predict
    predicted = neural_net.predict(X_test)
    error = loss(predicted, y_test)
    print(error)
    assert error < 1e-1
