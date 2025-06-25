import numpy as np
from scipy.optimize import minimize

###############################
# Gaussian Bias Potential
###############################

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

###############################
# Traditional ABC Implementation
###############################

import numpy as np
from scipy.optimize import minimize

class TraditionalABC:
    def __init__(
        self,
        potential,
        default_bias_height=10.0,
        default_bias_sigma=0.3, 
        optimizer='L-BFGS-B',
        starting_position=None
    ):
        self.potential = potential
        self.default_bias_height = default_bias_height
        self.default_bias_sigma = default_bias_sigma
        self.optimizer = optimizer
        
        self.reset(starting_position)
        
    def reset(self, starting_position=None):
        if starting_position is None:
            self.position = self.potential.default_starting_position()
        else:
            self.position = np.array(starting_position, dtype=float)
            
        self.bias_list = []
        self.trajectory = [self.position.copy()]  # All positions including descent steps
        self.energies = []  # unbiased energies at each trajectory point
        self.most_recent_hessian = None # store most recent approximate hessian from calculations for use in code and debugging
        self.forces = []  # Forces at each trajectory point
        self.minima = []
        self._dimension = len(self.position)
        self.total_force_calls = 0
        self.iter_periods = []
        
    @property
    def dimension(self):
        return self._dimension
        
    def total_potential(self, position):
        V = self.potential.potential(position)
        for bias in self.bias_list:
            V += bias.potential(position)
        return V
        
    def compute_force(self, position, eps=1e-5):
        """
        Compute total force due to biased PES at the given position.

        Uses analytical gradients if available, otherwise falls back to finite difference.

        Parameters:
        -----------
        position : array_like
            Position vector at which to compute force.
        eps : float, optional
            Finite difference step size (default: 1e-5).

        Returns:
        --------
        force : ndarray
            Force vector at the given position, clipped between -100 and 100.
        """
        try:
            total_grad = self.potential.gradient(position)
            for bias in self.bias_list:
                total_grad += bias.gradient(position)
            force = -total_grad
        except NotImplementedError:
            # fallback finite difference
            force = np.zeros_like(position)
            for i in range(len(position)):
                pos_plus = position.copy()
                pos_minus = position.copy()
                pos_plus[i] += eps
                pos_minus[i] -= eps
                force[i] = -(self.total_potential(pos_plus).item() - self.total_potential(pos_minus).item()) / (2 * eps)
            self.total_force_calls += 1

        return np.clip(force, -100, 100)
    
    def evaluate_critical_point(self, position, hessian):
        """
        Classify the critical point at the given position based on the Hessian.

        Parameters:
        -----------
        position : ndarray, shape (d,)
            Position of the critical point (unused here, but might be helpful elsewhere).
        hessian : ndarray, shape (d, d)
            Hessian matrix at the critical point.

        Returns:
        --------
        classification : str
            One of "minimum", "maximum", or "saddle".
        """
        eigvals = np.linalg.eigvalsh(hessian)  # Since Hessian is symmetric
        if np.all(eigvals > 0):
            return "minimum"
        elif np.all(eigvals < 0):
            return "maximum"
        else:
            return "saddle"

        
    def deposit_bias(self, center=None, covariance=None, height=None):
        pos=center if center is not None else self.position.copy()
        cov = covariance if covariance is not None else np.square(self.default_bias_sigma)
        h = height if height is not None else self.default_bias_height

        bias = GaussianBias(
            center=pos,
            covariance=cov, 
            height=h
        )
        self.bias_list.append(bias)
        print(f"Deposited bias at {pos} with (co)variance {cov} and height {h}.")
        
    def descend(self, max_steps, convergence_threshold=1e-5):
        # Record initial state before descent
        self.forces.append(self.compute_force(self.position))
        self.energies.append(self.total_potential(self.position))
        
        # Callback to record each optimization step
        def callback(xk):
            self.trajectory.append(xk.copy())
            self.forces.append(self.compute_force(xk))
            self.energies.append(self.potential.potential(xk))
            
        result = minimize(
            self.total_potential,
            self.position,
            method=self.optimizer,
            jac=lambda x: -self.compute_force(x),
            tol=convergence_threshold,
            options={'maxiter': max_steps, 'disp': False},
            callback=callback
        )

        # Ensure final state is recorded (in case callback missed it)
        if len(self.trajectory) == 0 or not np.allclose(self.trajectory[-1], result.x):
            self.trajectory.append(result.x.copy())
            self.forces.append(self.compute_force(result.x))
            self.energies.append(self.potential.potential(result.x))

        # if self.optimizer == "BFGS"
            # Get approximate hessian info from BFGS, store hessian from final step
        # elif "L-BFGS" in self.optimizer: 
            # Get whatever approximate info you can from LBFGS
        # else:
            # same as before, 
            
        # Update system state
        new_pos = result.x
        dist = np.linalg.norm(new_pos - self.trajectory[-2]) if len(self.trajectory) > 1 else 0
        
        
        if result.success:
            # If converged to near-0 force, check if also near-0 force on original PES 
            # reconstructing with cheap gradient calls instead of additional MD force calls
            unbiased_pes_force = self.forces[-1]
            for bias in self.bias_list:
                unbiased_pes_force -= (-bias.gradient(new_pos))

            if np.isclose(np.linalg.norm(unbiased_pes_force), 0, convergence_threshold):
                # final check: if hessian info available, check if the hessian of the original PES indicates a minimum
                if self.most_recent_hessian: 
                    unbiased_pes_hessian = self.most_recent_hessian
                    for bias in self.bias_list:
                        unbiased_pes_hessian -= bias.hessian(new_pos)

                    crit_type = self.evaluate_critical_point(new_pos, unbiased_pes_hessian)
                    if crit_type == "minimum": 
                        self.minima.append(new_pos.copy())
                    # If the hessian indicates a saddle point, you got quite lucky - better save it for future reference
                    elif crit_type == "saddle":
                        self.saddles.append(new_pos.copy())
                else: 
                    self.minima.append(new_pos.copy())

            
        self.position = new_pos
        return result.success

    def perturb(self, scale=0.05):
        noise = np.random.normal(scale=scale, size=self.position.shape)
        self.position += noise
        # self.trajectory.append(self.position.copy())
        # self.forces.append(self.compute_force(self.position))
        # self.energies.append(self.total_potential(self.position))
        print(f"Randomly perturbed to {self.position}")
    
    def store_iter_period(self):
        period = len(self.trajectory) - np.sum(self.iter_periods)
        self.iter_periods.append(period)

    def run_simulation(
        self,
        max_iterations=30,
        descent_max_steps=100,
        descent_threshold=1e-5,
        perturb_scale=0.05,
        verbose=True
    ):
        for iteration in range(max_iterations):
            converged = self.descend(
                max_steps=descent_max_steps,
                convergence_threshold=descent_threshold
            )

            self.deposit_bias()

            if converged:
                self.perturb(perturb_scale)
            
            self.store_iter_period()

            if verbose:
                print(f"Iteration {iteration+1}/{max_iterations}: "
                      f"Position {self.position}, "
                      f"Total biases: {len(self.bias_list)}")
                      
        print(f"Simulation completed. Total steps: {len(self.trajectory)}")
        
    def get_trajectory(self):
        return np.array(self.trajectory)
        
    def get_forces(self):
        return np.array(self.forces)
        
    def get_energies(self):
        return np.array(self.energies)
        
    def get_bias_centers(self):
        return np.array([bias.center for bias in self.bias_list])
        
    def compute_free_energy_surface(self, resolution=100):
        if self.dimension == 1:
            x_range = self.potential.plot_range()
            x = np.linspace(x_range[0], x_range[1], resolution)
            F = np.array([self.total_potential(np.array([xi])) for xi in x])
            return x, F
        elif self.dimension == 2:
            x_range, y_range = self.potential.plot_range()
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x, y)
            F = np.zeros_like(X)
            for i in range(resolution):
                for j in range(resolution):
                    pos = np.array([X[i,j], Y[i,j]])
                    F[i,j] = self.total_potential(pos)
            return (X, Y), F
        else:
            raise NotImplementedError("Visualization not implemented for dimensions > 2")


from deprecated.analysis_deprecated import plot_results, analyze_basin_visits
from potentials import DoubleWell1D, StandardMullerBrown2D, Complex1D

def run_1d_simulation():
    """Run 1D ABC simulation with double well potential."""
    np.random.seed(42)
    print("Starting 1D ABC Simulation")
    print("=" * 50)
    
    potential = Complex1D()
    abc = TraditionalABC(
        potential=potential,
        default_bias_height=1,
        default_bias_sigma=0.3,
        starting_position=[-1.2],
        optimizer="CG"
    )
    
    abc.run_simulation(max_iterations=80, perturb_scale=0.02, verbose=True)
    
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total steps: {len(trajectory)}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: {abc.position}")
    
    analyze_basin_visits(abc)
    plot_results(abc, save_plots=True, filename="1d_trad_abc.png")

def run_2d_simulation():
    """Run 2D ABC simulation with Muller-Brown potential."""
    np.random.seed(420)
    print("Starting 2D ABC Simulation")
    print("=" * 50)
    
    potential = StandardMullerBrown2D()
    abc = TraditionalABC(
        potential=potential,
        default_bias_height=10,
        default_bias_sigma=2,
        starting_position=[0, 0],
        optimizer="CG"
    )
    
    abc.run_simulation(max_iterations=80, perturb_scale=0.01, verbose=True)
    
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total steps: {len(trajectory)}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: {abc.position}")
    
    analyze_basin_visits(abc)
    plot_results(abc, save_plots=True, filename="2d_trad_abc.png")

def main():
    """Run both 1D and 2D simulations."""
    print("Running 1D Simulation")
    run_1d_simulation()
    
    # print("\nRunning 2D Simulation")
    # run_2d_simulation()

if __name__ == "__main__":
    main()