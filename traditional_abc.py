import numpy as np
from scipy.optimize import minimize

from potentials import DoubleWellPotential1D, MullerBrownPotential2D

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
    
    def evaluate(self, position):
        """
        Evaluate bias potential at given position(s).
        
        Parameters:
        -----------
        position : ndarray, shape (..., d)
            Positions at which to evaluate bias.
        
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
        bias_height=10.0,
        bias_sigma=0.3, 
        basin_radius=0.2,
        optimizer='L-BFGS-B',
        starting_position=None
    ):
        self.potential = potential
        self.bias_height = bias_height
        self.bias_sigma = bias_sigma
        self.basin_radius = basin_radius
        self.optimizer = optimizer
        
        self.reset(starting_position)
        
    def reset(self, starting_position=None):
        if starting_position is None:
            self.position = self.potential.default_starting_position()
        else:
            self.position = np.array(starting_position, dtype=float)
            
        self.bias_list = []
        self.trajectory = [self.position.copy()]  # All positions including descent steps
        self.forces = []  # Forces at each trajectory point
        self.energies = []  # Biased energies at each trajectory point
        self.minima = []
        self._dimension = len(self.position)
        
    @property
    def dimension(self):
        return self._dimension
        
    def total_potential(self, position):
        V = self.potential.potential(position)
        for bias in self.bias_list:
            V += bias.evaluate(position)
        return V
        
    def compute_force(self, position, eps=1e-5):
        force = np.zeros_like(position)
        for i in range(len(position)):
            pos_plus = position.copy()
            pos_minus = position.copy()
            pos_plus[i] += eps
            pos_minus[i] -= eps
            force[i] = -(self.total_potential(pos_plus) - self.total_potential(pos_minus)) / (2 * eps)
        return np.clip(force, -100, 100)
        
    def deposit_bias(self):
        bias = GaussianBias(
            center=self.position.copy(),
            covariance=np.square(self.bias_sigma),
            height=self.bias_height
        )
        self.bias_list.append(bias)
        print(f"Deposited bias at {self.position}")
        
    def descend(self, max_steps, convergence_threshold=1e-5):
        # Record initial state before descent
        self.forces.append(self.compute_force(self.position))
        self.energies.append(self.total_potential(self.position))
        
        # Callback to record each optimization step
        def callback(xk):
            self.trajectory.append(xk.copy())
            self.forces.append(self.compute_force(xk))
            self.energies.append(self.total_potential(xk))
            
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
            self.energies.append(self.total_potential(result.x))
            
        # Update system state
        new_pos = result.x
        dist = np.linalg.norm(new_pos - self.trajectory[-2]) if len(self.trajectory) > 1 else 0
        
        if result.success and dist > self.basin_radius:
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


from analysis import plot_results, analyze_basin_visits

def run_1d_simulation():
    """Run 1D ABC simulation with double well potential."""
    np.random.seed(42)
    print("Starting 1D ABC Simulation")
    print("=" * 50)
    
    potential = DoubleWellPotential1D()
    abc = TraditionalABC(
        potential=potential,
        bias_height=2,
        bias_sigma=0.5,
        basin_radius=0.5,
        starting_position=[-1.2]
    )
    
    abc.run_simulation(max_iterations=8, verbose=True)
    
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total steps: {len(trajectory)}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: {abc.position}")
    
    analyze_basin_visits(abc)
    plot_results(abc, save_plots=True, filename="1d_trad_abc.png")

def run_2d_simulation():
    """Run 2D ABC simulation with Muller-Brown potential."""
    np.random.seed(42)
    print("Starting 2D ABC Simulation")
    print("=" * 50)
    
    potential = MullerBrownPotential2D()
    abc = TraditionalABC(
        potential=potential,
        bias_height=20,
        bias_sigma=1,
        basin_radius=0.5,
        starting_position=[0, 0]
    )
    
    abc.run_simulation(max_iterations=15, perturb_scale=0.01, verbose=True)
    
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total steps: {len(trajectory)}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: {abc.position}")
    
    analyze_basin_visits(abc)
    plot_results(abc, save_plots=True, filename="2d_trad_abc.png")

def main():
    """Run both 1D and 2D simulations."""
    # print("Running 1D Simulation")
    # run_1d_simulation()
    
    print("\nRunning 2D Simulation")
    run_2d_simulation()

if __name__ == "__main__":
    main()