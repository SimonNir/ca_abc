import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import copy

###############################
# Potential Functions (1D)
###############################

def double_well(x):
    """Compute double well potential with minima at x=-1 and x=1."""
    return 1/6 * (5 * (x**2 - 1))**2

def gaussian_bias_1d(x, center, sigma, height):
    """Compute 1D Gaussian bias potential."""
    dx = x - center
    exponent = -dx**2 / (2 * sigma**2)
    exponent = np.clip(exponent, -100, 100)
    return height * np.exp(exponent)

def compute_force_1d(x, bias_list, eps=1e-5):
    """Compute negative gradient of total potential (force) in 1D."""
    def total_potential(pos_x):
        V = double_well(pos_x)
        for center, sigma, height in bias_list:
            V += gaussian_bias_1d(pos_x, center, sigma, height)
        return V
    
    # Numerical gradient
    V_x_plus = total_potential(x + eps)
    V_x_minus = total_potential(x - eps)
    
    dV_dx = (V_x_plus - V_x_minus) / (2 * eps)
    force = -dV_dx
    return np.clip(force, -100, 100)

###############################
# Standard ABC Implementation (1D)
###############################

class StandardABC1D:
    def __init__(self, dt=0.01, gamma=1.0, T=0.1, bias_height=10.0, 
                 bias_sigma=0.3, deposition_frequency=100, basin_threshold=0.1):
        """
        1D Standard ABC implementation.
        
        Parameters:
        -----------
        dt : float
            Time step for integration
        gamma : float
            Friction coefficient
        T : float
            Temperature (controls noise)
        bias_height : float
            Height of deposited Gaussian bias potentials
        bias_sigma : float
            Width (standard deviation) of Gaussian bias potentials
        deposition_frequency : int
            Number of MD steps between bias depositions
        basin_threshold : float
            Threshold for detecting if particle is in same basin
        """
        self.dt = dt
        self.gamma = gamma
        self.T = T
        self.bias_height = bias_height
        self.bias_sigma = bias_sigma
        self.deposition_frequency = deposition_frequency
        self.basin_threshold = basin_threshold
        
        # Noise parameters for Langevin dynamics
        self.noise_std = np.sqrt(2 * T * dt / gamma)
        
        # State variables
        self.position = np.array(0.0)  # Start at origin
        self.bias_list = []  # List of (center, sigma, height) tuples
        self.trajectory = []
        self.step_count = 0
        self.last_bias_position = None
        
    def reset(self, start_pos=None):
        """Reset the simulation."""
        if start_pos is None:
            self.position = np.array(0.0)
        else:
            self.position = np.array(start_pos)
        self.bias_list = []
        self.trajectory = [self.position.copy()]
        self.step_count = 0
        self.last_bias_position = None
        
    def should_deposit_bias(self):
        """
        Determine if a bias should be deposited.
        Standard ABC deposits bias periodically and when particle 
        has moved sufficiently from last bias position.
        """
        # Check frequency condition
        if self.step_count % self.deposition_frequency != 0:
            return False
            
        # Check if we've moved sufficiently from last bias
        if self.last_bias_position is not None:
            distance = np.abs(self.position - self.last_bias_position)
            if distance < self.basin_threshold:
                return False
                
        return True
        
    def deposit_bias(self):
        """Deposit a Gaussian bias at current position."""
        center = self.position.copy()
        self.bias_list.append((center, self.bias_sigma, self.bias_height))
        self.last_bias_position = center.copy()
        print(f"Step {self.step_count}: Deposited bias at {center:.3f}")
        
    def langevin_step(self):
        """Perform one step of Langevin dynamics."""
        # Compute force from potential + biases
        force = compute_force_1d(self.position, self.bias_list)
        
        # Add thermal noise
        noise = np.random.normal(0, self.noise_std)
        
        # Update position using Langevin equation
        self.position += (force / self.gamma) * self.dt + noise
        
        # Keep position in reasonable bounds
        self.position = np.clip(self.position, -2, 2)
        
    def run_simulation(self, max_steps=10000, verbose=True):
        """
        Run the ABC simulation.
        
        Parameters:
        -----------
        max_steps : int
            Maximum number of simulation steps
        verbose : bool
            Whether to print progress information
        """
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
                print(f"Step {step}: Position {self.position:.3f}, "
                      f"Biases deposited: {len(self.bias_list)}")
                      
        print(f"Simulation completed. Total biases deposited: {len(self.bias_list)}")
        
    def get_trajectory(self):
        """Return the trajectory as a numpy array."""
        return np.array(self.trajectory)
        
    def get_bias_centers(self):
        """Return the centers of all deposited biases."""
        if not self.bias_list:
            return np.array([])
        return np.array([bias[0] for bias in self.bias_list])
        
    def compute_free_energy_surface(self, x_range=(-2, 2), resolution=100):
        """
        Compute the free energy surface including biases.
        
        Returns:
        --------
        X : ndarray
            Coordinate values
        F : ndarray
            Free energy values
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        
        F = np.zeros_like(x)
        for i in range(resolution):
            V = double_well(x[i])
            for center, sigma, height in self.bias_list:
                V += gaussian_bias_1d(x[i], center, sigma, height)
            F[i] = V
                
        return x, F

###############################
# Analysis and Visualization (1D)
###############################

def plot_results_1d(abc_sim, save_plots=False):
    """Plot the results of 1D ABC simulation."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Original and biased potential
    x, F_orig = abc_sim.compute_free_energy_surface()
    ax1.plot(x, double_well(x), 'k-', label='Original Potential')
    
    # Plot biased potential
    x_bias, F_bias = abc_sim.compute_free_energy_surface()
    ax1.plot(x_bias, F_bias, 'b-', alpha=0.7, label='Biased Potential')
    
    # Mark bias centers
    bias_centers = abc_sim.get_bias_centers()
    if len(bias_centers) > 0:
        for center in bias_centers:
            ax1.axvline(center, color='r', linestyle='--', alpha=0.5)
    
    ax1.set_title('Potential Energy Landscape')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Energy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Trajectory
    trajectory = abc_sim.get_trajectory()
    times = np.arange(len(trajectory))
    ax2.plot(times, trajectory, 'g-', alpha=0.7, linewidth=1)
    ax2.axhline(-1, color='k', linestyle=':', alpha=0.3)  # Mark minima
    ax2.axhline(1, color='k', linestyle=':', alpha=0.3)
    
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
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('abc_1d_results.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_basin_visits_1d(trajectory, basin_centers=None, basin_radius=0.3):
    """
    Analyze which basins were visited during the simulation (1D version).
    
    Parameters:
    -----------
    trajectory : ndarray
        Trajectory of the simulation
    basin_centers : list
        Known basin centers in the potential
    basin_radius : float
        Radius to consider for basin occupancy
    """
    # Default basin centers for double well potential
    if basin_centers is None:
        basin_centers = [-1.0, 1.0]
    
    basin_visits = {i: [] for i in range(len(basin_centers))}
    current_basin = None
    
    for step, pos in enumerate(trajectory):
        # Find which basin we're in
        in_basin = None
        for j, center in enumerate(basin_centers):
            distance = np.abs(pos - center)
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
            print(f"Basin {i} (center: {basin_centers[i]}): {len(visits)} visits")
            print(f"  First visit at step: {visits[0]}")
            print(f"  Last visit at step: {visits[-1]}")
        else:
            print(f"Basin {i} (center: {basin_centers[i]}): Never visited")
    
    total_visits = sum(len(visits) for visits in basin_visits.values())
    print(f"\nTotal basin transitions: {total_visits}")
    
    return basin_visits

###############################
# Main Execution
###############################

def main_1d():
    """Run 1D ABC simulation and analysis."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Starting 1D ABC Simulation")
    print("=" * 50)
    
    # Initialize ABC with standard parameters
    abc = StandardABC1D(
        dt=0.001,
        gamma=1.0,
        T=1,
        bias_height=.2,
        bias_sigma=0.1,
        deposition_frequency=100,
        basin_threshold=0.1
    )
    
    # Run simulation
    abc.run_simulation(max_steps=10000, verbose=True)
    
    # Analyze results
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total steps: {len(trajectory)}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: {abc.position:.3f}")
    
    # Analyze basin exploration
    analyze_basin_visits_1d(trajectory)
    
    # Plot results
    plot_results_1d(abc, save_plots=True)
    
    return abc

if __name__ == "__main__":
    abc_simulation_1d = main_1d()