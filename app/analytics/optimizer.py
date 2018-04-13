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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str({
            'fun': self.fun,
            'x': self.x,
            'jac': self.jac,
            'message': self.message,
            'nfev': self.nfev,
            'nit': self.nit,
            'status': self.status,
            'success': self.success
        })


class Optimizer(metaclass=ABCMeta):
    def __init__(self, options: Dict):
        self.options = options
        self.res = Result()

    @abstractmethod
    def minimize(self, objective: Callable[[np.ndarray], float], init: np.ndarray) -> Dict:
        pass

    @staticmethod
    def numerical_gradient(theta: np.ndarray, function: Callable[[np.ndarray], float]) -> np.ndarray:
        '''Finite Differences of order 2, centered scheme'''
        epsilon = 1e-4
        n_parameters = theta.shape[0]
        delta = np.zeros(n_parameters)
        Jup = np.zeros(n_parameters)
        Jdown = np.zeros(n_parameters)
        for index in range(n_parameters):
            delta[index] = epsilon
            print(theta, type(theta))
            print(delta, type(delta))
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
        learning_rate = self.options.get('learning_rate', 1.0)
        maxiter = self.options.get('maxiter', 500)
        tol = self.options.get('tol', 1e-9)
        jac = self.options.get('jac', False)
        theta = init

        l2GradDelta = 0.0
        if not jac:
            def obj(theta: np.array) -> float:
                return objective(theta)[0]

        for iter_counter in range(maxiter):
            [cost, grad] = objective(theta)
            if not jac:
                grad = Optimizer.numerical_gradient(theta, obj)
            theta -= learning_rate * grad
            l2GradDelta = np.sum(grad * grad)
            if l2GradDelta < tol:
                self.res.message = 'optim successful'
                self.res.success = True
                break

        self.res.fun = cost
        self.res.jac = l2GradDelta
        self.res.nit = iter_counter
        self.res.x = theta
        return self.res


class LBFGSB(Optimizer):
    def __init__(self, options: Dict = {}):
        super(LBFGSB, self).__init__(options)

    def minimize(self, objective: Callable[[np.ndarray], Tuple[float, np.ndarray]], init: np.ndarray) -> Dict:
        def obj(theta: np.ndarray) -> float:
            return objective(theta)[0]

        def obj_grad(theta: np.ndarray) -> np.ndarray:
            return objective(theta)[1]

        jac = self.options.get('jac')
        try:
            del self.options['jac']
        except:
            print('options do not contain key `jac`')
        if jac:
            # jacobian provided
            res = scioptim.minimize(
                fun=obj,
                jac=obj_grad,
                x0=init,
                method='L-BFGS-B',
                options=self.options
            )
        else:
            # jacobian not provided, computed numerically
            res = scioptim.minimize(
                fun=obj,
                x0=init,
                method='L-BFGS-B',
                options=self.options
            )
        return res
