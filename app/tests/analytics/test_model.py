import numpy as np

from sklearn.model_selection import train_test_split

from analytics.model import NeuralNet
from analytics.optimizer import GradientDescent, LBFGSB, Optimizer
from analytics.util import Loss
from data.load import Load


def test_init():
    # nn arxi should be a list of dimentions or tuples
    arxitecture = [2, 2]
    nn = NeuralNet(arxitecture)
    assert True


def test_predict():
    # Set up of NN
    features, labels = Load.labelled_xnor()
    arxitecture = [2, 2, 1]
    nn = NeuralNet(arxitecture)
    # In Lieu of training
    nn.theta = np.array([-30, 20, 20, 10, -20, -20, -10, 20, 20])
    nn.thetas = []
    nn._decompose(nn.theta, nn.thetas)
    # Predict
    predictions = nn.predict(features)
    for prediction, label in zip(predictions, labels):
        print(prediction, label)
    assert Loss.accuracy(predictions, labels) > 0.99


def test_back_propagate():
    # Set-up
    layers = [2, 2, 1]
    neural_net = NeuralNet(layers)
    neural_net.theta = np.array([-30, 20, 20, 10, -20, -20, -10, 20, 20])
    neural_net.thetas = []
    neural_net._decompose(neural_net.theta, neural_net.thetas)
    loss = Loss.crossentropy
    # Backprop
    m = 100
    features, labels = Load.labelled_xnor(sample_size = m)
    predictions = neural_net._feed_forward(features)
    dJ, dJs = neural_net._back_propagate(predictions, labels)

    # numerical_gradient
    def objective(theta):
        nn = NeuralNet(layers)
        nn.theta = theta
        nn.thetas = []
        nn._decompose(nn.theta, nn.thetas)
        predicted = nn._feed_forward(features)
        return loss(predicted[-1][1], labels)
    num_dJ = Optimizer.numerical_gradient(neural_net.theta, objective)

    for nDjDtheta, DjDtheta in zip(num_dJ, dJ):
        print(nDjDtheta, DjDtheta, np.abs(nDjDtheta - DjDtheta))
        assert np.abs(nDjDtheta - DjDtheta) < 1e-4


def test_fit():
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
    error = loss(predicted, y_test)
    print(error)
    assert error < 1e-1


def test_decompose():
    arxitecture = [2, 2, 1]
    nn = NeuralNet(arxitecture)
    theta = np.array([-30, 20, 20, 10, -20, -20, -10, 20, 20])
    thetas = []
    nn._decompose(theta, thetas)
    assert len(thetas) == 2
    assert thetas[0].shape == (2, 3)
    assert thetas[1].shape == (1, 3)
    assert thetas[0][1, 0] == 10
    theta[3] = -10
    assert thetas[0][1, 0] == -10
    thetas[0][1, 0] = 20
    assert theta[3] == 20
