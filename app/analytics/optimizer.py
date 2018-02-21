import numpy as np
import scipy.optimize as scioptim

from abc import ABCMeta, abstractmethod
from typing import Dict, Callable, Tuple


class Result:
    def __init__(self):
        self.fun = 0.0
        self.x = np.empty((0,))
        self.jac = np.empty((0,))
        self.message = ''
        self.nfev = 0
        self.nit = 0
        self.status = 0
        self.success = False

    def __dict__(self):
        return {
            'fun': self.fun,
            'x': self.x,
            'jac': self.jac,
            'message': self.message,
            'nfev': self.nfev,
            'nit': self.nit,
            'status': self.status,
            'success': self.success
        }


class Optimizer(metaclass=ABCMeta):
    def __init__(self, options: Dict):
        self.options = options
        self.res = Result()

    @abstractmethod
    def minimize(self, objective: Callable[[np.ndarray], float], init: np.ndarray) -> Dict:
        pass

    @staticmethod
    def numerical_gradient(theta: np.ndarray, function: Callable[[np.ndarray], float]) -> np.ndarray:
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
        super(GradientDescent, self).__init__(options)

    def minimize(self, objective: Callable[[np.ndarray], Tuple[float, np.ndarray]], init: np.ndarray) -> Dict:
        '''
        This function performs a simple gradient descent
        '''
        learningRate = self.options.get('learning_rate', 1.0)
        maxiter = self.options.get('maxiter', 500)
        tol = self.options.get('tol', 1e-9)
        jac = self.options.get('jac', False)
        theta = init

        l2GradDelta = 0.0

        for iterCounter in range(maxiter):
            if jac:
                [cost, grad] = objective(theta)
            else:
                cost = objective(theta)
                grad = Optimizer.numerical_gradient(theta, objective)
            theta -= learningRate * grad
            l2GradDelta = np.sum(grad * grad)
            if l2GradDelta < tol:
                self.res.message = 'optim successful'
                self.res.success = True
                break

        self.res.fun = cost
        self.res.jac = l2GradDelta
        self.res.nit = iterCounter
        self.res.x = theta
        return self.res


class LBFGSB(Optimizer):
    def __init__(self, options: Dict = {}):
        super(LBFGSB, self).__init__(options)

    def minimize(self, objective: Callable[[np.ndarray], Tuple[float, np.ndarray]], init: np.ndarray) -> Dict:
        res = scioptim.minimize(
            fun=objective,
            x0=init,
            method='L-BFGS-B',
            options=self.options
        )
        return res

# Deprecated
# ===========

class BSGD(Optimizer):
    # TODO this does not works
    def __init__(self, options: Dict):
        super(BSGD, self).__init__(options)

    def minimize(self, objective: Callable[[np.ndarray], float],
                 init: np.ndarray) -> Dict:
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
        batchSize = int(floor(m / 2.0))
        # end specfic

        for iterCounter in range(maxiter):
            # below only specific to SGD
            batchIndex = sample(range(m), batchSize)
            [cost, grad] = objective(theta, features[batchIndex],
                                     labels[batchIndex])
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
