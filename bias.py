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
    
    def gradient(self, position):
        """
        Compute the gradient of the Gaussian bias potential at given position(s).

        Parameters:
        -----------
        position : ndarray, shape (..., d)
            Positions at which to compute gradient.
        
        Returns:
        --------
        grad : ndarray, shape (..., d)
            Gradient(s) of the bias potential.
        """
        pos = np.atleast_2d(position)
        delta = pos - self.center
        bias = self.potential(pos)[:, np.newaxis]  # shape (N, 1)
        grad = -bias * np.dot(delta, self._cov_inv.T)  # shape (N, d)
        return grad if position.ndim > 1 else grad[0]
    
    def hessian(self, position):
        """
        Compute the Hessian of the Gaussian bias potential at given position(s).

        Parameters:
        -----------
        position : ndarray, shape (..., d)
            Positions at which to compute the Hessian.
        
        Returns:
        --------
        hess : ndarray, shape (..., d, d)
            Hessian(s) of the bias potential.
        """
        pos = np.atleast_2d(position)  # shape (N, d)
        delta = pos - self.center      # shape (N, d)
        bias = self.potential(pos)     # shape (N,)
        
        # Precompute inverse covariance
        cov_inv = self._cov_inv        # shape (d, d)

        hess_list = []
        for i in range(pos.shape[0]):
            delta_i = delta[i][:, np.newaxis]  # shape (d, 1)
            outer = cov_inv @ delta_i @ delta_i.T @ cov_inv  # shape (d, d)
            hess_i = bias[i] * (outer - cov_inv)             # shape (d, d)
            hess_list.append(hess_i)

        hess_array = np.stack(hess_list)  # shape (N, d, d)
        return hess_array if position.ndim > 1 else hess_array[0]
    
    def get_cholesky(self):
        """Return the Cholesky decomposition of covariance matrix."""
        return np.linalg.cholesky(self.covariance)
    
    def __repr__(self):
        return (f"GaussianBias(center={self.center}, covariance=\n{self.covariance}, height={self.height})")