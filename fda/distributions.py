from typing import Callable
import numpy as np
from .fdata import FData

class GaussianProcess:
    r"""Class to simulate Gaussian processes

    Keyword arguments:
        mean -- FData that is the mean of the process
        covariance -- function k(s,t) \in \mathbb{R} such k(s,t) = Cov(X_s, X_t)
        alpha -- regularization
    """
    def __init__(self, mean: FData, covariance: Callable[[float, float], float], alpha: float = 0.1):
        self.grid = mean.grid
        self.mean = mean
        self.covariance = covariance
        self.alpha = alpha
    def generateSamples(self, N=100):
        """ Generate N samples of the process"""
        d = len(self.grid)
        K = np.zeros(shape=(d,d))
        mean = self.mean.value
        for i in range(d):
            for j in range(d):
                if i < j:
                    K[i, j] = self.covariance(self.grid[i], self.grid[j])
                    K[j, i] = K[i, j]
                elif i == j:
                    K[i, i] = 1 + self.alpha
        normalSamples = np.random.multivariate_normal(mean, K, N)
        
        return [FData(normalSamples[m,:], self.grid) for m in range(N)]