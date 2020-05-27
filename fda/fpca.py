import numpy as np
from .fdata import mean, FData, scprod

class FPCA:
    """Functional Principal Components 
    """
    def __init__(self, k=2, regularization=0.01, centerData=False):
        """Functional Principal Components Analysis

        Keyword Arguments:
            k {int} -- Number of components (default: {2})
            regularization {float} -- Regularization to apply to the diagonal (default: {0.01})
            centerData {bool} -- Whether it should center the data (default: {False})
        """
        self.k = k
        self.centerData = centerData
        self.alpha = regularization

    def centerData_(self, curves):
        """Centers the data

        Arguments:
            curves {Iterable[FData]} -- Samples

        Returns:
            Iterable[FData] -- Centered samples or the same samples
        """
        if self.centerData:
            mu = mean(curves)
            centered = [f - mu for f in curves]
            self.mu = mu
        else:
            centered = [f for f in curves]
        return centered
        

    def fit(self, curves):
        """Fits FPCA to data

        Arguments:
            curves {Iterable[FData]} -- Data

        Returns:
            ({numpy.array}, {Iterable[FData]}) -- Eigenvalues and Eigenfunctions
        """
        # Center data if necessary
        centered = self.centerData_(curves)
        
        # We get the time grid for first sample
        t = curves[0].grid
        # Create a 2D time grid
        T1, T2 = np.meshgrid(t,t)
        # V will be our discrete covariance operator
        V = np.zeros((t.shape[0], t.shape[0]))
        # Pre-compute the function values
        interpolated =[[f(s) for s in t] for f in centered]
        for ii, _ in enumerate(T1[0,:]):
            for jj, _ in enumerate(T2[0,:]):
                V[ii, jj] = sum([f[ii] * f[jj] for f in interpolated])
                if ii == jj:
                    V[ii, jj] += self.alpha #regularization
                
        # Solve the linear system to obtain the eigenvalues
        eigenvalues, eigenfunctions = np.linalg.eigh(V)
        # Transform eigenfunctions to FData format
        eigenfunctions = eigenfunctions[:,np.argsort(-eigenvalues*eigenvalues)]
        eigenfunctions = [FData(eigenfunctions[:,i].T) for i in range(eigenvalues.shape[0])]
        # Store the data
        self.eigenvalues, self.eigenfunctions = eigenvalues, eigenfunctions
        
        return eigenvalues, eigenfunctions
    
    def transform(self, curves):
        """Computes the embedding representation and compressed curves

        Arguments:
            curves {Iterable[FData]} -- Samples

        Returns:
            ({numpy.array, Iterable[FData]}) -- Coefficient and compressed curves
        """
        # Center the data
        centered = self.centerData_(curves)
        # Obtain the coefficients in the embedding
        coefficients = [[scprod(f, eigenfunction) for eigenfunction in self.eigenfunctions[:self.k]] for f in centered]
        coefficients = np.asarray(coefficients)
        # Obtain the projected function
        projected = [sum([c * phi for c, phi in zip(row, self.eigenfunctions)]) for row in coefficients]
        return coefficients, projected

if __name__ == "__main__":
    x = np.linspace(0, 1, 100)
    curves = [FData(np.sin(x), x) for _ in range(20)]
    fpca = FPCA()
    fpca.fit(curves)
    coefficients, projected = fpca.transform(curves)