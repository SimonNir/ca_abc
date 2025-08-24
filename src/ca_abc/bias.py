import numpy as np 

class GaussianBias:
    """
    N-dimensional Gaussian bias potential. Can be defined over a full space
    or a specific subspace of degrees of freedom.

    V(x) = height * exp( -0.5 * (x_sub - center_sub)^T @ cov_inv_sub @ (x_sub - center_sub) )

    Parameters:
    -----------
    center : ndarray, shape (d_full,)
        Center of the Gaussian in the full coordinate space.
    covariance : ndarray, shape (d_sub, d_sub) or float
        Covariance matrix of the Gaussian. If dof_indices is specified, this
        is the reduced covariance for that subspace.
    height : float
        Height (amplitude) of the Gaussian bias.
    dof_indices : list or ndarray, optional
        The indices of the degrees of freedom this bias applies to. If None,
        the bias is assumed to apply to all degrees of freedom.
    """

    def __init__(self, center, covariance, height, dof_indices=None):
        self.center = np.atleast_1d(center)
        self.height = height
        self.dof_indices = dof_indices
        
        # The stored covariance is always the (potentially reduced) matrix
        if np.isscalar(covariance):
            dim = len(dof_indices) if dof_indices is not None else len(center)
            self.covariance = np.eye(dim) * covariance
        else:
            self.covariance = np.atleast_2d(covariance)

        # Dimension validation
        expected_dim = len(self.dof_indices) if self.dof_indices is not None else len(self.center)
        if self.covariance.shape[0] != expected_dim:
            raise ValueError(f"Covariance dimension ({self.covariance.shape[0]}) does not match "
                             f"expected dimension ({expected_dim}) based on dof_indices.")

        if self.covariance.shape[0] != self.covariance.shape[1]:
            raise ValueError(f"Covariance matrix must be square. Currently:\n{self.covariance}")

        # Pre-compute the inverse of the (reduced) covariance matrix
        try:
            # Check for positive definiteness and compute inverse
            np.linalg.cholesky(self.covariance)
            self._cov_inv = np.linalg.inv(self.covariance)
        except np.linalg.LinAlgError:
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
        return (f"GaussianBias(height={self.height}, dof_indices={self.dof_indices}, "
                f"covariance=\n{self.covariance}, \ncenter=\n{self.center})")