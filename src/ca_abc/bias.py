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
        
        if np.isscalar(covariance):
            self.covariance = np.eye(len(center)) * covariance
        else:
            self.covariance = np.atleast_2d(covariance)
        
        if self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError(f"Covariance matrix must be square. Currently:\n{self.covariance}")
        if self.covariance.shape[0] != self.center.shape[0]:
            raise ValueError("Covariance matrix dimension must match center dimension")
        
        # Check positive definiteness by Cholesky (raises if not PD)
        try:
            self._cholesky = np.linalg.cholesky(self.covariance)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix must be positive definite")
        
        # Compute inverse
        self._cov_inv = np.linalg.inv(self.covariance)
        
        # Compute determinant from Cholesky factor: det = (prod(diag(L)))^2
        diag_L = np.diag(self._cholesky)
        log_det = 2.0 * np.sum(np.log(diag_L))
        self._det_cov = np.clip(np.exp(log_det), 1e-100, 1e100)  # will be 0 if underflow, but no exception raised here

        if not np.isfinite(log_det) or log_det == -np.inf:
            raise ValueError("Covariance matrix determinant log is invalid, matrix may be singular or ill-conditioned")


    
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
        pos = np.atleast_2d(position)
        delta = pos - self.center       # shape (N, d)
        bias = self.potential(pos)      # shape (N,)

        # Efficient vectorized Hessian
        outer = np.einsum('ni,nj->nij', delta @ self._cov_inv, delta @ self._cov_inv)
        hess = bias[:, None, None] * (outer - self._cov_inv)
        return hess if position.ndim > 1 else hess[0]
    
    def get_cholesky(self):
        """Return the Cholesky decomposition of covariance matrix."""
        return self._cholesky
    
    def __repr__(self):
        return (f"GaussianBias(center={self.center}, covariance=\n{self.covariance}, height={self.height})")