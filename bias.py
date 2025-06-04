import numpy as np 

class GaussianBias:
    """
    N-dimensional Gaussian bias potential:
    
    V(x) = -height * exp( -0.5 * (x - center)^T @ cov_inv @ (x - center) )
    
    Parameters:
    -----------
    center : ndarray, shape (d,)
        Center of the Gaussian.
    covariance : ndarray, shape (d, d) or float
        Covariance matrix of the Gaussian (must be positive definite) or scalar for isotropic Gaussian.
    height : float
        Height (amplitude) of the Gaussian bias.
    """
    
    def __init__(self, center, covariance, height):
        self.center = np.atleast_1d(center)
        self.height = height
        
        # Handle scalar covariance input
        if np.isscalar(covariance):
            self.covariance = np.eye(len(center)) * covariance**2
        else:
            self.covariance = np.atleast_2d(covariance)
        
        # Validate covariance matrix
        if self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError("Covariance matrix must be square")
        if self.covariance.shape[0] != self.center.shape[0]:
            raise ValueError("Covariance matrix dimension must match center dimension")
        
        # Compute inverse and determinant for efficient evaluation
        self._cov_inv = np.linalg.inv(self.covariance)
        self._det_cov = np.linalg.det(self.covariance)
        if self._det_cov <= 0:
            raise ValueError("Covariance matrix must be positive definite")
    
    def potential(self, position):
        """
        Apply bias potential at given position(s).
        
        Parameters:
        -----------
        position : ndarray, shape (..., d)
            Positions at which to apply bias.
        
        Returns:
        --------
        bias : ndarray, shape (...)
            Bias potential value(s).
        """
        pos = np.atleast_2d(position)
        delta = pos - self.center
        exponent = -0.5 * np.einsum('ij,jk,ik->i', delta, self._cov_inv, delta)
        bias = self.height * np.exp(exponent)
        return bias if position.ndim > 1 else bias[0]