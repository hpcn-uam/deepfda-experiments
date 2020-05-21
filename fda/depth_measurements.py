import numpy as np

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