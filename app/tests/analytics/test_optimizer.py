import numpy as np

from typing import Tuple
from analytics.util import Loss
from analytics.optimizer import LBFGSB, GradientDescent, Optimizer


def linear_regression(features, theta):
    return theta[1] * features + theta[0]


def test_numerical_gradient():
    def linear(x: np.ndarray) -> float:
        return np.sum(x)

    def square(x: np.ndarray) -> float:
        return np.sum(x * x) / 2.0

    def cube(x: np.ndarray) -> float:
        return np.sum(x * x * x) / 3.0

    x = np.array([-1.0, 0.0, 1.0, 2.0, 10.0])
    DlinearDx = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    DsquareDx = np.array([-1.0, 0.0, 1.0, 2.0, 10.0])
    DcubeDx = np.array([1.0, 0.0, 1.0, 4.0, 100.0])

    tolerance = 1e-8
    for sut, expected in zip(
            Optimizer.numerical_gradient(x, linear), DlinearDx):
        assert np.abs(sut - expected) < tolerance
    for sut, expected in zip(
            Optimizer.numerical_gradient(x, square), DsquareDx):
        assert np.abs(sut - expected) < tolerance
    for sut, expected in zip(
            Optimizer.numerical_gradient(x, cube), DcubeDx):
        assert np.abs(sut - expected) < tolerance


def test_gradientdescent():
    n_observations = 100
    theta_seeked = np.array([10.0, 1.0])
    features = np.random.rand(n_observations)
    labels = linear_regression(features, theta_seeked)
    options = {
        'learningRate': 1.0,
        'maxiter': 500,
        'tol': 1e-9,
        'jac': False
    }

    def objective(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        predicted = linear_regression(features, theta)
        loss = Loss.squared_error(predicted, labels)
        return loss, None

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
    labels = linear_regression(features, theta_seeked)
    options = {}

    def objective(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        predicted = linear_regression(features, theta)
        loss = Loss.squared_error(predicted, labels)
        return loss, None

    theta0 = np.random.rand(2)
    optimizer = LBFGSB(options)
    sut = optimizer.minimize(objective, theta0)

    print(sut)
    assert sut.success
    n_param = theta_seeked.shape[0]
    for i in range(n_param):
        assert abs(theta_seeked[i] - sut.x[i]) < 1e-6
