import numpy as np
import scipy.optimize as scioptim

from abc import ABCMeta, abstractmethod
from typing import Dict, Callable, List


class Optimizer(ABCMeta):
    def __init__(self, options: Dict):
        self.options = options
        pass

    @abstractmethod
    def minimize(self, objective: Callable[[float], np.ndarray], init: np.ndarray) -> Dict:
        pass

    @staticmethod
    def numerical_gradient(theta: np.ndarray, function: Callable[[float], np.ndarray]) -> np.ndarray:
        epsilon = 1e-4
        n_parameters = theta.shape[0]
        delta = np.zeros(n_parameters)
        Jup = np.zeros(n_parameters)
        Jdown = np.zeros(n_parameters)
        for index in range(n_parameters):
            delta[index] = epsilon
            Jup[index] = function(theta + delta)
            Jdown[index] = function(theta - delta)
            delta[index] = 0.0
        return 0.5 * (Jup - Jdown) / epsilon


class GradientDescent(Optimizer):
    def __init__(self, options: Dict):
        super(GradientDescent).__init__(options)

    def minimize(self, objective: Callable[[float], List[np.ndarray]], init: np.ndarray) -> Dict:
        '''
        This function performs a simple gradient descent
        '''
        learningRate = self.options['learningRate']
        maxiter = self.options['maxiter']
        tol = self.options['tol']
        theta = init

        res = {}
        res['success'] = False
        res['message'] = 'max iteration reached'

        l2GradDelta = 0.0

        for iterCounter in range(maxiter):
            [cost, grad] = objective(theta)
            theta -= learningRate * grad
            l2GradDelta = np.sum(grad * grad)
            if l2GradDelta < tol:
                res['message'] = 'optim successful'
                res['success'] = True
                break

        res['fun'] = cost
        res['funDelta'] = l2GradDelta
        res['numIter'] = iterCounter
        return res


class BSGD(Optimizer):
    def __init__(self, options: Dict):
        super(BSGD).__init__(options)

    def minimize(self, objective: Callable[[float], List[np.ndarray]], init: np.ndarray) -> Dict:
        '''
        This function performs a batch stochastic gradient descent
        --> will not work as is for now
        '''
        learningRate = self.options['learningRate']
        maxiter = self.options['maxiter']
        tol = self.options['tol']
        theta = init

        res = {}
        res['success'] = False
        res['message'] = 'max iteration reached'

        # below only specific to SGD
        l2GradDelta = 0.0
        m = features.__len__()
        batchSize = int(floor(m/2.0))
        # end specfic

        for iterCounter in range(maxiter):
            # below only specific to SGD
            batchIndex = sample(range(m), batchSize)
            [cost, grad] = objective(theta, features[batchIndex], labels[batchIndex])
            # end specfic
            theta -= learningRate * grad
            l2GradDelta = np.sum(grad * grad)
            if l2GradDelta < tol:
                res['message'] = 'optim successful'
                res['success'] = True
                break

        res['fun'] = cost
        res['funDelta'] = l2GradDelta
        res['numIter'] = iterCounter
        return res


class LBFGSB(Optimizer):
    def __init__(self, options: Dict):
        LBFGSB(BSGD).__init__(options)

    def minimize(self, objective: Callable[[float], List[np.ndarray]], init: np.ndarray) -> Dict:
        res = scioptim.minimize(
            fun=objective,
            x0=init,
            method='L-BFGS-B',
            options=self.options,
            jac=True
        )
        return res
