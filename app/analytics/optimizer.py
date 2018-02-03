import numpy as np
import scipy.optimize as scioptim

from abc import ABCMeta, abstractmethod
from typing import Dict, Callable


class Optimizer(ABCMeta):
    def __init__(self, options: Dict):
        self.options = options
        pass

    @abstractmethod
    def minimize(self, objective: Callable[[float], [np.array]], init: np.array) -> Dict:
        pass


class GradientDescent(Optimizer):
    def __init__(self, options: Dict):
        super(GradientDescent).__init__(options)

    def minimize(self, objective: Callable[[float], [np.array, np.array]], init: np.array) -> Dict:
        '''
        This function performs a simple gradient descent
        '''
        learningRate = self.options['learningRate']
        maxiter = self.options['maxiter']
        tol = self.options['tol']
        vectorTheta = init

        res = {}
        res['success'] = False
        res['message'] = 'max iteration reached'

        l2GradDelta = 0.0

        for iterCounter in range(maxiter):
            [cost, grad] = objective(vectorTheta)
            vectorTheta -= learningRate * grad
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

    def minimize(self, objective: Callable[[float], [np.array, np.array]], init: np.array) -> Dict:
        '''
        This function performs a batch stochastic gradient descent
        --> will not work as is for now
        '''
        learningRate = self.options['learningRate']
        maxiter = self.options['maxiter']
        tol = self.options['tol']
        vectorTheta = init

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
            [cost, grad] = objective(vectorTheta, features[batchIndex], labels[batchIndex])
            # end specfic
            vectorTheta -= learningRate * grad
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

    def minimize(self, objective: Callable[[float], [np.array, np.array]], init: np.array) -> Dict:
        res = scioptim.minimize(
            fun=objective,
            x0=init,
            method='L-BFGS-B',
            options=self.options,
            jac=True
        )
        return res
