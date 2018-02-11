import numpy as np

from analytics.util import Loss
from analytics.optimizer import LBFGSB, GradientDescent


def linear(features, theta):
    return theta[1] * features + theta[0]


def test_gradientdescent():
    n_observations = 100
    theta_seeked = np.array([10.0, 1.0])
    features = np.random.rand(n_observations)
    labels = linear(features, theta_seeked)
    options = {
        'learningRate': 1.0,
        'maxiter': 500,
        'tol': 1e-9,
        'jac': False
    }

    def objective(theta):
        predicted = linear(features, theta)
        loss = Loss.squared_error(predicted, labels)
        return loss

    theta0 = np.random.rand(2)
    optimizer = GradientDescent(options)
    sut = optimizer.minimize(objective, theta0)

    print(sut)
    assert sut.success
    n_param = theta_seeked.shape[0]
    for i in range(n_param):
        assert abs(theta_seeked[i] - sut.x[i]) < 1e-3


def test_LBFGSB():
    n_observations = 100
    theta_seeked = np.array([10.0, 1.0])
    features = np.random.rand(n_observations)
    labels = linear(features, theta_seeked)
    options = {}

    def objective(theta):
        predicted = linear(features, theta)
        loss = Loss.squared_error(predicted, labels)
        return loss

    theta0 = np.random.rand(2)
    optimizer = LBFGSB(options)
    sut = optimizer.minimize(objective, theta0)

    print(sut)
    assert sut.success
    n_param = theta_seeked.shape[0]
    for i in range(n_param):
        assert abs(theta_seeked[i] - sut.x[i]) < 1e-6
