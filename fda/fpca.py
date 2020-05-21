import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import matplotlib
from typing import Callable

class FData:
    def __init__(self, Y, X=None):
        self.value = Y
        if X is None:
            X = np.arange(0, len(Y))
        self.grid = X

    def __add__(self, other):
        if type(other) is FData:
            other = other.value
        return FData(self.value + other, X=self.grid)
    __radd__ = __add__
    
    def __sub__(self, other):
        if type(other) is FData:
            other = other.value
        return FData(self.value - other, X=self.grid)

    
    def __mul__(self, other):
        if type(other) is FData:
            other = other.value
        return FData(self.value * other, X=self.grid)
    __rmul__ = __mul__
    
    def __abs__(self):
        return FData(np.abs(self.value), X=self.grid)

    def __truediv__(self, other):
        if type(other) is FData:
            other = other.value
        return FData(self.value / other, X=self.grid)

    def __str__(self):
        return self.value.__str__()
    
    def __call__(self, x):
        return np.interp(np.asarray(x), np.asarray(self.grid), np.asarray(self.value))
        
    def plt(self, **kwargs):
        if self.grid is None:
            return plt.plot(self.value, **kwargs)

        return plt.plot(self.grid, self.value, **kwargs)

    def integrate(self):
        return spi.simps(self.value, x=self.grid)
        
    def L1norm(self):
        return spi.simps(np.abs(self.value), x=self.grid)
    
    def L2norm(self):
        return np.sqrt(spi.simps(np.multiply(self.value, self.value), x=self.grid))

def L2distance(X,Y):
    #print(X, Y)
    return (X-Y).L2norm()

def L1distance(X,Y):
    #print(X, Y)
    return (X-Y).L1norm()


def mean(array):
    if len(array) > 0:
        s = None
        for curve in array:
            if s is None:
                s = curve
            else:
                s = s + curve

        return s / len(array)
    else:
        return np.NaN

def percentile(array, p=50):
    if len(array) > 0:
        matrix = np.array([x.value for x in array])
        
        return FData(np.percentile(matrix, p, axis=0))
    else:
        return np.NaN
    
def quantile(array, p=0.5):
    if len(array) > 0:
        matrix = np.array([x.value for x in array])
        
        return FData(np.quantile(matrix, p, axis=0))
    else:
        return np.NaN
    
    
def median(array):
    if len(array) > 0:
        matrix = np.array([x.value for x in array])
        
        return FData(np.median(matrix, axis=0), array[0].grid)
    else:
        return np.NaN
        
def scprod(x,y):
    return (x*y).integrate()

class GaussianProcess:
    def __init__(self, mean: FData, covariance: Callable[[float, float], float], alpha = 0.1):
        self.grid = mean.grid
        self.mean = mean
        self.covariance = covariance
        self.alpha = alpha
    def generateSamples(self, N=100):
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
    
def hipoGraphPoints(f,g):
    """ 
        Devuelve los puntos en los que f(x) > g(x) y una mascara
    """
    grid = f.grid
    return grid[f>g], f>g


def epiGraphPoints(f,g):
    """ 
        Devuelve los puntos en los que f(x) < g(x) y una mascara
    """
    grid = f.grid
    return grid[f<g], f<g


def SL(f, array):
    return np.mean([np.sum(f.value <= g.value) / len(f.grid) for g in array])

def IL(f, array):
    return np.mean([np.sum(f.value >= g.value) / len(f.grid) for g in array])

def MS(f, array):
    return min(SL(f, array), IL(f, array))

def deepest(array, depth=MS):
    return array[np.argmax([depth(f, array) for f in array])]

    
class KMeans:
    def __init__(self, k, distance=L2distance, depth_f=None):
        self.k = k
        self.distance = distance
        self.depth_function = depth_f
        if self.depth_function is None:
            self.depth_function = median
    
    
    def move_centroids(self, curves, closest, centroids):
        """returns the new centroids assigned from the curves closest to them"""
        res = []
        for k, _ in enumerate(centroids):
            filtered_curves = [curve for i, (curve, closest) in enumerate(zip(curves, closest)) if closest == k]
            res.append(self.depth_function(filtered_curves))
        return res
        
    def closest_centroid(self, curves, centroids=None, distance=L2distance):
        """returns an array containing the index to the nearest centroid for each point"""
        if centroids is None:
            centroids = self.centroids
        distances = [[self.distance(curve, centroid) for centroid in centroids] for curve in curves]
        return np.argmin(distances, axis=1)

    def initialize_centroids_random(self, curves):
        """returns k centroids from the initial points"""
        centroids = curves.copy()
        np.random.shuffle(centroids)
        return centroids[:self.k]
    
    def initialize_centroids(self, curves):
        """returns k centroids from the initial points"""
        for i in range(self.k):
            if i == 0:
                centroids = [curves[0]]
            else:
                distances = [[self.distance(curve, centroid) for centroid in centroids] for curve in curves]
                minD = np.min(distances, axis=1)
                weights = minD / np.sum(minD)
                centroid = np.random.choice(curves, 1, p = weights)[0]
                centroids.append(centroid)
        return centroids
    
    def train(self, curves, iters=100):
        self.centroids = self.initialize_centroids(curves)
        self.continue_training(curves, iters=iters)
        
    def silhouette_a(self, curves, i):
        closest = np.array(self.closest_centroid(curves))
        cluster_of_i = closest[i]
        size_of_cluster_i = np.sum(closest == cluster_of_i)
        i = curves[i]

        s = 0
        for j, cluster_of_j in zip(curves, closest):
            if cluster_of_j == cluster_of_i:
                s += self.distance(i,j)
        return (1/(size_of_cluster_i - 1)) * s
    
    def silhouette_b(self, curves, i):
        closest = np.array(self.closest_centroid(curves))
        cluster_of_i = closest[i]
        curve_i = curves[i]
        arr = []
        for k in np.unique(closest):
            if k == cluster_of_i:
                continue
            size_of_cluster_k = np.sum(closest == k)
            s = 0
            for j, cluster_of_j in zip(curves, closest):
                if cluster_of_j == k:
                    s += self.distance(curve_i, j)
            arr.append(1/size_of_cluster_k * s)
        return np.min(arr)
    
    def silhouette_s(self, curves, i):
        a = self.silhouette_a(curves, i)
        b = self.silhouette_b(curves, i)
        if a <= b:
            return 1 - a/b
        else:
            return b/a - 1
    
    def silhouette_coefficient(self, curves):
        return np.mean([self.silhouette_s(curves, i) for i, _ in enumerate(curves)])
        
            
    def continue_training(self, curves, iters=100):
        for _ in range(iters):
            closest = self.closest_centroid(curves, self.centroids)
            self.centroids = self.move_centroids(curves, closest, self.centroids)
            
    def plotFunctionalClusters(self, curves, cm):
        closests = self.closest_centroid(curves, self.centroids)
        if type(cm) is matplotlib.colors.LinearSegmentedColormap:
            cm = cm(np.linspace(0,cm.N-1, len(self.centroids)).astype(int))
        for i, centroid in enumerate(self.centroids):
            color = cm[i]
            centroid.plt(color=color, linewidth=5)
            for curve, closest in zip(curves, closests):
                if closest == i:
                    curve.plt(color=color, alpha=0.5, linestyle="dashed")
