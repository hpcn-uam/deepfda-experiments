from .distances import L2distance
from .fdata import median
import numpy as np
import matplotlib
    
class KMeans:
    def __init__(self, k, distance=L2distance, depth_f=median):
        self.k = k
        self.distance = distance
        self.depth_function = depth_f
    
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
