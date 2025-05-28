import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

###############################
# Base Potential Class (Abstract)
###############################

class PotentialEnergySurface(ABC):
    """Abstract base class for potential energy surfaces."""
    
    @abstractmethod
    def potential(self, position):
        """Compute potential energy at given position."""
        pass
        
    @abstractmethod
    def default_starting_position(self):
        """Return default starting position for this PES."""
        pass
        
    @abstractmethod
    def plot_range(self):
        """Return plotting range for visualization."""
        pass
        
    @abstractmethod
    def known_basins(self):
        """Return known basins (for analysis)."""
        pass

###############################
# Concrete Potential Implementations
###############################

class DoubleWellPotential1D(PotentialEnergySurface):
    """1D double well potential."""
    
    def potential(self, x):
        """Compute double well potential with minima at x=-1 and x=1."""
        return 1/6 * (5 * (x**2 - 1))**2
        
    def default_starting_position(self):
        return np.array([-1.0], dtype=float)
        
    def plot_range(self):
        return (-2, 2)
        
    def known_basins(self):
        return [np.array([-1.0], dtype=float), np.array([1.0], dtype=float)]

class MullerBrownPotential2D(PotentialEnergySurface):
    """2D Muller-Brown potential."""
    
    def potential(self, pos):
        """Compute the Muller-Brown potential with numerical safeguards."""
        x, y = pos[0], pos[1]
        A = np.array([-200, -100, -170, 15])
        a = np.array([-1, -1, -6.5, 0.7])
        b = np.array([0, 0, 11, 0.6])
        c = np.array([-10, -10, -6.5, 0.7])
        x0 = np.array([1, 0, -0.5, -1])
        y0 = np.array([0, 0.5, 1.5, 1])
        
        V = 0.0
        for i in range(4):
            exponent = a[i]*(x - x0[i])**2 + b[i]*(x - x0[i])*(y - y0[i]) + c[i]*(y - y0[i])**2
            exponent = np.clip(exponent, -100, 100)
            V += A[i] * np.exp(exponent)
        return V
        
    def default_starting_position(self):
        return np.array([0.0, 0.0], dtype=float)
        
    def plot_range(self):
        return ((-2, 2), (-1, 2))
        
    def known_basins(self):
        return [
            np.array([-0.558, 1.442]),  # Basin A
            np.array([0.623, 0.028]),   # Basin B  
            np.array([-0.050, 0.467])   # Basin C
        ]

###############################
# Gaussian Bias Potential
###############################

class GaussianBias:
    """N-dimensional Gaussian bias potential."""
    
    def __init__(self, center, sigma, height):
        """
        Initialize Gaussian bias.
        
        Parameters:
        -----------
        center : ndarray
            Center position of the Gaussian
        sigma : float or ndarray
            Width(s) of the Gaussian (can be scalar or per-dimension)
        height : float
            Height of the Gaussian
        """
        self.center = np.array(center)
        self.sigma = sigma
        self.height = height
        
    def evaluate(self, position):
        """Compute bias potential at given position."""
        delta = position - self.center
        if np.isscalar(self.sigma):
            r_sq = np.sum(delta**2) / (2 * self.sigma**2)
        else:
            r_sq = np.sum((delta / self.sigma)**2) / 2
            
        exponent = np.clip(-r_sq, -100, 100)
        return self.height * np.exp(exponent)

###############################
# Generalized ABC Implementation
###############################

class LangevinABC:
    def __init__(self, potential, dt=0.01, gamma=1.0, T=0.1, 
                 bias_height=10.0, bias_sigma=0.3, 
                 deposition_frequency=100, basin_threshold=0.1,
                 force_threshold=None, starting_position=None):
        """
        Generalized ABC implementation for N-dimensional PES.
        
        Parameters:
        -----------
        potential : PotentialEnergySurface
            The potential energy surface to simulate
        dt : float
            Time step for integration
        gamma : float
            Friction coefficient
        T : float
            Temperature (controls noise)
        bias_height : float
            Height of deposited Gaussian bias potentials
        bias_sigma : float or ndarray
            Width(s) of Gaussian bias potentials (scalar or per-dimension)
        deposition_frequency : int
            Number of MD steps between bias depositions
        basin_threshold : float
            Threshold for detecting if particle is in same basin
        force_threshold : float or None
            Threshold for force magnitude to determine if climbing
        starting_position : ndarray or None
            Starting position for simulation
        """
        self.potential = potential
        self.dt = dt
        self.gamma = gamma
        self.T = T
        self.bias_height = bias_height
        self.bias_sigma = bias_sigma
        self.deposition_frequency = deposition_frequency
        self.basin_threshold = basin_threshold
        self.force_threshold = force_threshold
        
        # State variables
        if starting_position is None:
            self.position = potential.default_starting_position()
        else:
            self.position = np.array(starting_position, dtype=float)
            
        self.bias_list = []  # List of GaussianBias objects
        self.trajectory = [self.position.copy()]
        self.step_count = 0
        self.last_bias_position = None
        self.dimension = len(self.position)
        
    def reset(self, start_pos=None):
        """Reset the simulation."""
        if start_pos is None:
            self.position = self.potential.default_starting_position()
        else:
            self.position = np.array(start_pos)
        self.bias_list = []
        self.trajectory = [self.position.copy()]
        self.step_count = 0
        self.last_bias_position = None
        
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
            # Create shifted positions
            pos_plus = position.copy()
            pos_minus = position.copy()
            pos_plus[i] += eps
            pos_minus[i] -= eps
            
            # Numerical gradient
            V_plus = self.total_potential(pos_plus)
            V_minus = self.total_potential(pos_minus)
            force[i] = -(V_plus - V_minus) / (2 * eps)
            
        return np.clip(force, -100, 100)
        
    def should_deposit_bias(self):
        """Determine if a bias should be deposited."""
        # Frequency condition
        if self.step_count % self.deposition_frequency != 0:
            return False
            
        # Force condition
        if self.force_threshold is not None:
            force = self.compute_force(self.position)
            if np.linalg.norm(force) > self.force_threshold:
                return False
                
        # # Distance condition
        # if self.last_bias_position is not None:
        #     distance = np.linalg.norm(self.position - self.last_bias_position)
        #     if distance < self.basin_threshold:
        #         return False
                
        return True
        
    def deposit_bias(self):
        """Deposit a Gaussian bias at current position."""
        bias = GaussianBias(
            center=self.position.copy(),
            sigma=self.bias_sigma,
            height=self.bias_height
        )
        self.bias_list.append(bias)
        self.last_bias_position = self.position.copy()
        print(f"Step {self.step_count}: Deposited bias at {self.position}")
        
    def calculate_noise(self):
        """Calculate thermal noise for Langevin dynamics."""
        noise_std = np.sqrt(2 * self.T * self.dt / self.gamma)
        return np.random.normal(0, noise_std, size=self.dimension)
        
    def langevin_step(self):
        """Perform one step of Langevin dynamics."""
        # Compute force
        force = self.compute_force(self.position)
        
        # Add thermal noise
        noise = self.calculate_noise()
        
        # Update position
        self.position += (force / self.gamma) * self.dt + noise
        
        # Apply reasonable bounds (optional)
        if self.dimension == 1:
            self.position = np.clip(self.position, -2, 2)
        elif self.dimension == 2:
            self.position = np.clip(self.position, [-3, -1], [3, 2])
        
    def run_simulation(self, max_steps=10000, verbose=True):
        """Run the ABC simulation."""
        for step in range(max_steps):
            self.step_count = step
            
            # Check if we should deposit a bias
            if self.should_deposit_bias():
                self.deposit_bias()
                
            # Perform Langevin dynamics step
            self.langevin_step()
            
            # Record trajectory
            self.trajectory.append(self.position.copy())
            
            # Print progress
            if verbose and step % 1000 == 0:
                print(f"Step {step}: Position {self.position}, "
                      f"Biases deposited: {len(self.bias_list)}")
                      
        print(f"Simulation completed. Total biases deposited: {len(self.bias_list)}")
        
    def get_trajectory(self):
        """Return the trajectory as a numpy array."""
        return np.array(self.trajectory)
        
    def get_bias_centers(self):
        """Return the centers of all deposited biases."""
        if not self.bias_list:
            return np.zeros((0, self.dimension))
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

def plot_results(abc_sim, filename, save_plots=False):
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
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # plt.show()

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
    window_size = 500
    variances = []
    times = []
    
    for i in range(window_size, len(trajectory), 100):
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
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # plt.show()

def analyze_basin_visits(abc_sim, basin_radius=0.3):
    """
    Analyze which basins were visited during the simulation.
    
    Parameters:
    -----------
    abc_sim : GeneralizedABC
        The ABC simulation object
    basin_radius : float
        Radius to consider for basin occupancy
    """
    trajectory = abc_sim.get_trajectory()
    known_basins = abc_sim.potential.known_basins()
    
    basin_visits = {i: [] for i in range(len(known_basins))}
    current_basin = None
    
    for step, pos in enumerate(trajectory):
        # Find which basin we're in
        in_basin = None
        for j, center in enumerate(known_basins):
            distance = np.linalg.norm(pos - center)
            if distance < basin_radius:
                in_basin = j
                break
                
        # Record basin transitions
        if in_basin != current_basin and in_basin is not None:
            basin_visits[in_basin].append(step)
            current_basin = in_basin
    
    print("\nBasin Analysis:")
    print("-" * 40)
    for i, visits in basin_visits.items():
        if visits:
            print(f"Basin {i} (center: {known_basins[i]}): {len(visits)} visits")
            print(f"  First visit at step: {visits[0]}")
            print(f"  Last visit at step: {visits[-1]}")
        else:
            print(f"Basin {i} (center: {known_basins[i]}): Never visited")
    
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
    abc = LangevinABC(
        potential=potential,
        dt=0.001,
        gamma=1.0,
        T=1.0,
        bias_height=0.2,
        bias_sigma=0.1,
        deposition_frequency=100,
        basin_threshold=0.1,
        force_threshold=10,
        starting_position=[-1.0]
    )
    
    abc.run_simulation(max_steps=15000, verbose=True)
    
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total steps: {len(trajectory)}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: {abc.position}")
    
    analyze_basin_visits(abc)
    plot_results(abc, save_plots=True, filename="1d_lang_abc.png")

def run_2d_simulation():
    """Run 2D ABC simulation with Muller-Brown potential."""
    np.random.seed(42)
    print("Starting 2D ABC Simulation")
    print("=" * 50)
    
    potential = MullerBrownPotential2D()
    abc = LangevinABC(
        potential=potential,
        dt=0.001,
        gamma=1.0,
        T=1,
        bias_height=10.0,
        bias_sigma=0.3,
        deposition_frequency=100,
        basin_threshold=0.1,
        starting_position=[0, 0]
    )
    
    abc.run_simulation(max_steps=10000, verbose=True)
    
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total steps: {len(trajectory)}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: {abc.position}")
    
    analyze_basin_visits(abc)
    plot_results(abc, save_plots=True, filename="2d_lang_abc.png")

def run_2d_simulation_with_force_threshold():
    """Run 2D ABC simulation with Muller-Brown potential."""
    np.random.seed(42)
    print("Starting 2D ABC Simulation")
    print("=" * 50)
    
    potential = MullerBrownPotential2D()
    abc = LangevinABC(
        potential=potential,
        dt=0.001,
        gamma=1.0,
        T=0.1,
        bias_height=10.0,
        bias_sigma=0.3,
        deposition_frequency=1,
        basin_threshold=0.1,
        force_threshold=100,
        starting_position=[0, 0]
    )
    
    abc.run_simulation(max_steps=10000, verbose=True)
    
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total steps: {len(trajectory)}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: {abc.position}")
    
    analyze_basin_visits(abc)
    plot_results(abc, save_plots=True, filename="2d_lang_abc_force_thresh.png")

def main():
    """Run both 1D and 2D simulations."""
    print("Running 1D Simulation")
    run_1d_simulation()
    
    print("\nRunning 2D Simulation")
    run_2d_simulation()

    print("\nRunning 2D with force threshold")
    run_2d_simulation_with_force_threshold()

if __name__ == "__main__":
    main()