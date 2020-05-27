"""
FDA package. FData file

@author: Daniel Perdices <daniel.perdices@uam.es>
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

class FData:
    """
    Class for functional data 

    Parameters
    ----------
    Y : numpy array
        Values of the curve
    X : numpy array, optional
        Time grid. The default is None.
        If none is provided, asumes 0..len(Y)
    """
    def __init__(self, Y, X=None):
        """ Constructor """
        self.value = Y
        if X is None:
            X = np.arange(0, len(Y))
        self.grid = X

    def __add__(self, other):
        """ Addition operator """
        if type(other) is FData:
            other = other.value
        return FData(self.value + other, X=self.grid)
    __radd__ = __add__
    
    def __sub__(self, other):
        """ Substraction operator """
        if type(other) is FData:
            other = other.value
        return FData(self.value - other, X=self.grid)

    
    def __mul__(self, other):
        """ Multiplication operator """
        if type(other) is FData:
            other = other.value
        return FData(self.value * other, X=self.grid)
    __rmul__ = __mul__
    
    def __abs__(self):
        """ Abosulute value operator """
        return FData(np.abs(self.value), X=self.grid)

    def __truediv__(self, other):
        """ Division operator """
        if type(other) is FData:
            other = other.value
        return FData(self.value / other, X=self.grid)

    def __str__(self):
        """ Formatter operator """
        return self.value.__str__()
    
    def __call__(self, x):
        """ 
        Call operator.
        As a function, we use an interpolator to compute the value 
            
        """
        return np.interp(np.asarray(x), np.asarray(self.grid), np.asarray(self.value))
        
    def plt(self, **kwargs):
        """
        Plot the curve

        Parameters
        ----------
        **kwargs : dict
            Arguments for plt.plot.

        Returns
        -------
        matplotlib plot
            The handler of the figure.

        """
        if self.grid is None:
            return plt.plot(self.value, **kwargs)

        return plt.plot(self.grid, self.value, **kwargs)

    def integrate(self):
        """
        Computes the integral of the curve along the grid
        The integral of a curve is computed through Simpson's method

        Returns
        -------
        float
            The integral from min(grid) to max(grid).

        """
        return spi.simps(self.value, x=self.grid)
        
    def L1norm(self):
        """
        Computes the L1 norm of the curve

        Returns
        -------
        float
            The L1 norm.

        """
        return spi.simps(np.abs(self.value), x=self.grid)
    
    def L2norm(self):
        """
        Computes the L2 norm of the curve

        Returns
        -------
        float
            The L2 norm.

        """
        return np.sqrt(spi.simps(np.multiply(self.value, self.value), x=self.grid))


def mean(array):
    """
    Returns the mean of an iterable of curves

    Parameters
    ----------
    array : Iterable[FData]
        The list of curves.

    Returns
    -------
    FData
        The mean curve.

    """
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
    """
    Returns the percentile of an iterable of curves

    Parameters
    ----------
    array : Iterable[FData]
        The list of curves.
    p : float
        The percentile to be compute
    Returns
    -------
    FData
        The pth-percentile curve.

    """
    if len(array) > 0:
        matrix = np.array([x.value for x in array])
        
        return FData(np.percentile(matrix, p, axis=0))
    else:
        return np.NaN
    
def quantile(array, p=0.5):
    """
    Returns the quantile of an iterable of curves

    Parameters
    ----------
    array : Iterable[FData]
        The list of curves.
    p : float
        The quantile to be compute

    Returns
    -------
    FData
        The pth-percentile curve.

    """
    if len(array) > 0:
        matrix = np.array([x.value for x in array])
        
        return FData(np.quantile(matrix, p, axis=0))
    else:
        return np.NaN
    
    
def median(array):
    """
    Returns the median of an iterable of curves

    Parameters
    ----------
    array : Iterable[FData]
        The list of curves.

    Returns
    -------
    FData
        The median curve.

    """
    if len(array) > 0:
        matrix = np.array([x.value for x in array])
        
        return FData(np.median(matrix, axis=0), array[0].grid)
    else:
        return np.NaN
        
def scprod(x,y):
    """
    Computes the scalar product between X and Y

    Parameters
    ----------
    x : FData
        First operand to be multiplied.
    y : FData
        Second operand to be multiplied.

    Returns
    -------
    float
        The scalar product <X,Y>.

    """
    return (x*y).integrate()

def asMatrix(array):
    """
    Returns the grid representation of some functional data. Each row is a sample and each column 
    a value of the grid

    Parameters
    ----------
    array : Iterable of FData
        Data to be converted

    Returns
    -------
    numpy.matrix
        The matrix representation where each

    """

    return np.asmatrix([curve.value for curve in array])
