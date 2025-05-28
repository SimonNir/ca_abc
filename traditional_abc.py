import numpy as np
import matplotlib.pyplot as plt
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

class TraditionalABC:
    def __init__(self, potential, bias_height=10.0, bias_sigma=0.3, 
                 convergence_threshold=1e-5, starting_position=None, basin_radius=0.2, 
                 max_abc_iters=30, max_descent_steps=500, optimizer='L-BFGS-B'):
        """
        Simplified ABC implementation where each iteration is a full BFGS convergence.
        
        Parameters:
        -----------
        potential : PotentialEnergySurface
            The potential energy surface to simulate
        bias_height : float
            Height of deposited Gaussian bias potentials
        bias_sigma : float or ndarray
            Width(s) of Gaussian bias potentials (scalar or per-dimension)
        convergence_threshold : float
            Threshold for considering minimization converged
        starting_position : ndarray or None
            Starting position for simulation
        """
        self.potential = potential
        self.bias_height = bias_height
        self.bias_sigma = bias_sigma
        self.convergence_threshold = convergence_threshold
        self.basin_radius = basin_radius
        self.max_descent_steps = max_descent_steps 
        self.optimizer = optimizer
        
        # State variables
        if starting_position is None:
            self.position = potential.default_starting_position()
        else:
            self.position = np.array(starting_position, dtype=float)
            
        self.bias_list = []  # List of GaussianBias objects
        self.trajectory = [self.position.copy()]
        self.dimension = len(self.position)
        self.minima = []
        self.descend_step_list = []
        
    def reset(self, start_pos=None):
        """Reset the simulation."""
        if start_pos is None:
            self.position = self.potential.default_starting_position()
        else:
            self.position = np.array(start_pos)
        self.bias_list = []
        self.trajectory = [self.position.copy()]
        
    def total_potential(self, position):
        """Compute total potential (PES + biases)."""
        V = self.potential.potential(position)
        for bias in self.bias_list:
            V += bias.evaluate(position)
        return V
        
    def compute_force(self, position, eps=1e-5):
        """Compute negative gradient of total potential (force)."""
        force = np.zeros_like(position)
        for i in range(len(position)):
            pos_plus = position.copy()
            pos_minus = position.copy()
            pos_plus[i] += eps
            pos_minus[i] -= eps
            force[i] = -(self.total_potential(pos_plus) - self.total_potential(pos_minus)) / (2 * eps)
        return np.clip(force, -100, 100)
        
    def deposit_bias(self):
        """Deposit a Gaussian bias at current position."""
        bias = GaussianBias(
            center=self.position.copy(),
            covariance=np.square(self.bias_sigma),
            height=self.bias_height
        )
        self.bias_list.append(bias)
        print(f"Deposited bias at {self.position}")
        
    def descend(self):
        """Perform minimization until convergence or max steps hit (plateau-safe)."""
        result = minimize(
            self.total_potential,
            self.position,
            method=self.optimizer,
            jac=lambda x: -self.compute_force(x),
            tol=self.convergence_threshold,
            options={'maxiter': self.max_descent_steps, 'disp': False}
        )

        converged = result.success
        energy_calls = result.nfev
        force_calls = result.njev 
        self.descend_step_list.append((energy_calls, force_calls))

        new_pos = result.x
        dist = np.linalg.norm(new_pos - self.position)
        # Only add to minima if converged *and* moved significantly
        if converged:
            if dist > self.basin_radius:
                self.minima.append(new_pos.copy())
        # Always update position and trajectory
        self.position = new_pos 
        self.trajectory.append(self.position.copy())

        return converged

    def perturb(self, scale=0.05):
        """Apply a random perturbation to the current position."""
        noise = np.random.normal(scale=scale, size=self.position.shape)
        self.position += noise  
        
        
    def run_simulation(self, max_abc_iters=30, perturb_scale=0.05, verbose=True):
        """Run the ABC simulation for fixed number of full minimizations."""
        for iteration in range(max_abc_iters):
            # Perform full minimization
            converged = self.descend()

            # Always deposit after convergence
            self.deposit_bias()

            if converged:
                self.perturb(perturb_scale) 
            
            if verbose:
                print(f"Iteration {iteration+1}/{max_abc_iters}: "
                      f"Position {self.position}, "
                      f"Total biases: {len(self.bias_list)}")
                      
        print(f"Simulation completed. Total biases deposited: {len(self.bias_list)}")
        
    def get_trajectory(self):
        """Return the trajectory as a numpy array."""
        return np.array(self.trajectory)
        
    def get_bias_centers(self):
        """Return the centers of all deposited biases."""
        return np.array([bias.center for bias in self.bias_list])
        
    def compute_free_energy_surface(self, resolution=100):
        """
        Compute the free energy surface including biases.
        
        Returns:
        --------
        coords : ndarray or tuple of ndarrays
            Coordinate values (1D: array, 2D: meshgrid)
        F : ndarray
            Free energy values
        """
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

###############################
# Analysis and Visualization
###############################

def plot_results(abc_sim, filename=None, save_plots=False):
    """Plot the results of ABC simulation."""
    dimension = abc_sim.dimension
    
    if dimension == 1:
        _plot_1d_results(abc_sim, save_plots, filename)
    elif dimension == 2:
        _plot_2d_results(abc_sim, save_plots, filename)
    else:
        print(f"Visualization not supported for {dimension}D systems")

def _plot_1d_results(abc_sim, save_plots, filename="1d_results.png"):
    """Plot results for 1D system."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Original and biased potential
    x, F_orig = abc_sim.compute_free_energy_surface()
    ax1.plot(x, abc_sim.potential.potential(x), 'k-', label='Original Potential')
    ax1.plot(x, F_orig, 'b-', alpha=0.7, label='Biased Potential')
    
    # Mark bias centers
    bias_centers = abc_sim.get_bias_centers()
    if len(bias_centers) > 0:
        for center in bias_centers:
            ax1.axvline(center[0], color='r', linestyle='--', alpha=0.5)
    
    ax1.set_title('Potential Energy Landscape')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Energy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Trajectory
    trajectory = abc_sim.get_trajectory()
    times = np.arange(len(trajectory))
    ax2.plot(times, trajectory, 'g-', alpha=0.7, linewidth=1)
    
    # Mark known basins
    for basin in abc_sim.potential.known_basins():
        ax2.axhline(basin[0], color='k', linestyle=':', alpha=0.3)
    
    ax2.set_title('ABC Trajectory')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Histogram of visited positions
    ax3.hist(trajectory, bins=50, density=True, color='purple', alpha=0.7)
    ax3.set_title('Visited Positions Distribution')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(abc_sim.potential.plot_range())
    
    # Plot 4: Energy profile over time
    energy_profile = np.array([abc_sim.total_potential(pos) for pos in trajectory])
    ax4.plot(times, energy_profile, 'orange', linewidth=1)
    ax4.set_title('Energy Profile Over Time')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Potential Energy')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        if filename is None:
            filename = "1d_results.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def _plot_2d_results(abc_sim, save_plots, filename="2d_results.png"):
    """Plot results for 2D system."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Original potential
    (X, Y), _ = abc_sim.compute_free_energy_surface()
    Z_orig = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z_orig[i,j] = abc_sim.potential.potential(np.array([X[i,j], Y[i,j]]))
    
    def log_levels(Z, num): 
        Z_min = np.min(Z)
        Z_shifted = Z - Z_min + 1e-1
        log_levels = np.logspace(np.log10(1e-1), np.log10(np.max(Z_shifted)), num=num)
        return log_levels + Z_min
    
    contour1 = ax1.contour(X, Y, Z_orig, levels=log_levels(Z_orig, 50), colors='black', alpha=0.6)
    ax1.clabel(contour1, inline=True, fontsize=8)
    ax1.set_title('Original Potential')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Biased potential surface
    (X_bias, Y_bias), Z_bias = abc_sim.compute_free_energy_surface()
    contour2 = ax2.contour(X_bias, Y_bias, Z_bias, levels=log_levels(Z_orig, 50), colors='blue', alpha=0.6)
    ax2.clabel(contour2, inline=True, fontsize=8)
    
    # Overlay bias positions
    bias_centers = abc_sim.get_bias_centers()
    if len(bias_centers) > 0:
        ax2.scatter(bias_centers[:,0], bias_centers[:,1], 
                   c='red', s=50, marker='x', linewidths=2, label='Bias Centers')
        ax2.legend()
    
    ax2.set_title('Biased Potential Surface')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Trajectory
    from matplotlib.collections import LineCollection
    
    trajectory = abc_sim.get_trajectory()
    points = trajectory.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Time-based colormap
    t = np.linspace(0, len(trajectory), len(segments))
    lc = LineCollection(segments, cmap='viridis', array=t, linewidth=2, alpha=0.8)
    
    ax3.add_collection(lc)
    
    # Plot start and end points
    ax3.scatter(trajectory[0,0], trajectory[0,1], c='green', s=100, 
                marker='o', label='Start', zorder=5)
    ax3.scatter(trajectory[-1,0], trajectory[-1,1], c='red', s=100, 
                marker='s', label='End', zorder=5)
    
    # Plot bias centers if present
    if len(bias_centers) > 0:
        ax3.scatter(bias_centers[:,0], bias_centers[:,1], 
                    c='red', s=10, marker='x', linewidths=2, label='Biases', alpha=0.8, zorder=4)
    
    ax3.set_title('ABC Trajectory')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Colorbar for time
    cbar = plt.colorbar(lc, ax=ax3, orientation='vertical')
    cbar.set_label('Simulation Time Step')
    
    # Plot 4: Exploration metrics
    window_size = 2
    variances = []
    times = []
    
    for i in range(window_size, len(trajectory), 1):
        window = trajectory[i-window_size:i]
        var = np.sum(np.var(window, axis=0))  # Sum of variances in all dimensions
        variances.append(var)
        times.append(i)
    
    ax4.plot(times, variances, 'b-', linewidth=2)
    ax4.set_title('Exploration Measure (Position Variance)')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Total Variance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        if filename is None:
            filename = "2d_results.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_basin_visits(abc_sim, basin_radius=0.3, verbose=True):
    """
    Analyze which basins were visited during the simulation.
    
    Parameters:
    -----------
    abc_sim : TraditionalABC
        The ABC simulation object.
    basin_radius : float
        Radius to consider for basin occupancy.
    verbose : bool
        Whether to print detailed analysis.

    Returns:
    --------
    basin_visits : dict
        Mapping from basin index to list of step indices where it was visited.
    """
    trajectory = abc_sim.get_trajectory()
    known_basins = [np.atleast_1d(b) for b in abc_sim.potential.known_basins()]

    basin_visits = {i: [] for i in range(len(known_basins))}
    current_basin = None

    for step, pos in enumerate(trajectory):
        pos = np.atleast_1d(pos)
        in_basin = None
        for j, center in enumerate(known_basins):
            distance = np.linalg.norm(pos - center)
            if distance < basin_radius:
                in_basin = j
                break

        # Detect basin transition
        if in_basin != current_basin and in_basin is not None:
            basin_visits[in_basin].append(step)
            current_basin = in_basin

    if verbose:
        print("\nBasin Analysis:")
        print("-" * 40)
        for i, visits in basin_visits.items():
            center = known_basins[i]
            if visits:
                print(f"Basin {i} (center: {center}): {len(visits)} visit(s)")
                print(f"  First visit at step: {visits[0]}")
                print(f"  Last visit at step: {visits[-1]}")
            else:
                print(f"Basin {i} (center: {center}): Never visited")

        total_visits = sum(len(visits) for visits in basin_visits.values())
        print(f"\nTotal basin transitions: {total_visits}")

    return basin_visits


###############################
# Main Execution
###############################

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
        convergence_threshold=1e-5, 
        starting_position=[-1.2]
    )
    
    abc.run_simulation(max_abc_iters=8, verbose=True)
    
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total iterations: {len(trajectory)}")
    print(f"Total energy calls: {np.sum(np.array([call[0] for call in abc.descend_step_list]))}")
    print(f"Total force calls: {np.sum(np.array([call[1] for call in abc.descend_step_list]))}")
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
        bias_height=10,
        bias_sigma=0.5,
        basin_radius=.5,
        convergence_threshold=1e-5, 
        starting_position=[0, 0]
    )
    
    abc.run_simulation(max_abc_iters=15, perturb_scale=0.05, verbose=True)
    
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total iterations: {len(trajectory)}")
    print(f"Total energy calls: {np.sum(np.array([call[0] for call in abc.descend_step_list]))}")
    print(f"Total force calls: {np.sum(np.array([call[1] for call in abc.descend_step_list]))}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: {abc.position}")
    
    analyze_basin_visits(abc)
    plot_results(abc, save_plots=True, filename="2d_trad_abc.png")


def main():
    """Run both 1D and 2D simulations."""
    print("Running 1D Simulation")
    run_1d_simulation()
    
    print("\nRunning 2D Simulation")
    run_2d_simulation()

if __name__ == "__main__":
    main()