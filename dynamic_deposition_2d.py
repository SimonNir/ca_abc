import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import copy

###############################
# Potential Functions
###############################

def muller_brown(x, y):
    """Compute the Muller-Brown potential with numerical safeguards."""
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

def gaussian_bias(x, y, center, sigma, height):
    """Compute Gaussian bias potential."""
    cx, cy = center
    dx = x - cx
    dy = y - cy
    r_sq = dx**2 + dy**2
    exponent = -r_sq / (2 * sigma**2)
    exponent = np.clip(exponent, -100, 100)
    return height * np.exp(exponent)

def compute_force(x, y, bias_list, eps=1e-5):
    """Compute negative gradient of total potential (force)."""
    def total_potential(pos_x, pos_y):
        V = muller_brown(pos_x, pos_y)
        for center, sigma, height in bias_list:
            V += gaussian_bias(pos_x, pos_y, center, sigma, height)
        return V
    
    # Numerical gradient
    V_x_plus = total_potential(x + eps, y)
    V_x_minus = total_potential(x - eps, y)
    V_y_plus = total_potential(x, y + eps)
    V_y_minus = total_potential(x, y - eps)
    
    dV_dx = (V_x_plus - V_x_minus) / (2 * eps)
    dV_dy = (V_y_plus - V_y_minus) / (2 * eps)
    
    force = -np.array([dV_dx, dV_dy])
    return np.clip(force, -100, 100)

###############################
# Standard ABC Implementation
###############################

class StandardABC:
    def __init__(self, dt=0.01, gamma=1.0, T=0.1, bias_height=10.0, 
                 bias_sigma=0.3, deposition_frequency=100, basin_threshold=0.1, 
                 starting_position=[0,0], force_threshold=0.1):
        """
        Standard ABC implementation.
        
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
        self.force_threshold = force_threshold
        
        # Noise parameters for Langevin dynamics
        self.noise_std = np.sqrt(2 * T * dt / gamma)
        
        # State variables
        self.position = np.array(starting_position, dtype=np.float64)
        self.bias_list = []  # List of (center, sigma, height) tuples
        self.trajectory = [self.position.copy()]
        self.step_count = 0
        self.last_bias_position = None
        
    def reset(self, start_pos=None):
        """Reset the simulation."""
        if start_pos is None:
            self.position = np.array([0.0, 0.0])
        else:
            self.position = np.array(start_pos)
        self.bias_list = []
        self.trajectory = [self.position.copy()]
        self.step_count = 0
        self.last_bias_position = None
        
    def should_deposit_bias(self):
        forces = compute_force(self.position[0], self.position[1], self.bias_list)
        if np.linalg.norm(forces) > self.force_threshold:
            return False  # Still climbing
        if self.last_bias_position is not None:
            distance = np.linalg.norm(self.position - self.last_bias_position)
            if distance < self.basin_threshold:
                return False  # Not far enough
        return True
        
    def deposit_bias(self):
        """Deposit a Gaussian bias at current position."""
        center = self.position.copy()
        self.bias_list.append((center, self.bias_sigma, self.bias_height))
        self.last_bias_position = center.copy()
        print(f"Step {self.step_count}: Deposited bias at ({center[0]:.3f}, {center[1]:.3f})")
        
    def langevin_step(self):
        """Perform one step of Langevin dynamics."""
        # Compute force from potential + biases
        force = compute_force(self.position[0], self.position[1], self.bias_list)
                
        # Add thermal noise
        noise = np.random.normal(0, self.noise_std, size=2)
        
        # Update position using Langevin equation
        self.position += (force / self.gamma) * self.dt + noise
        
        # Keep position in reasonable bounds
        self.position = np.clip(self.position, -3, 3)
        
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
                print(f"Step {step}: Position ({self.position[0]:.3f}, {self.position[1]:.3f}), "
                      f"Biases deposited: {len(self.bias_list)}")
                      
        print(f"Simulation completed. Total biases deposited: {len(self.bias_list)}")
        
    def get_trajectory(self):
        """Return the trajectory as a numpy array."""
        return np.array(self.trajectory)
        
    def get_bias_centers(self):
        """Return the centers of all deposited biases."""
        if not self.bias_list:
            return np.array([]).reshape(0, 2)
        return np.array([bias[0] for bias in self.bias_list])
        
    def compute_free_energy_surface(self, x_range=(-2, 2), y_range=(-1, 2), resolution=100):
        """
        Compute the free energy surface including biases.
        
        Returns:
        --------
        X, Y : ndarray
            Meshgrid coordinates
        F : ndarray
            Free energy values
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        F = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                V = muller_brown(X[i, j], Y[i, j])
                for center, sigma, height in self.bias_list:
                    V += gaussian_bias(X[i, j], Y[i, j], center, sigma, height)
                F[i, j] = V
                
        return X, Y, F

###############################
# Analysis and Visualization
###############################

def plot_results(abc_sim, save_plots=True):
    """Plot the results of ABC simulation."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.2)
    
    # Plot 1: Original Muller-Brown potential
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z_orig = np.zeros_like(X)
    
    for i in range(100):
        for j in range(100):
            Z_orig[i, j] = muller_brown(X[i, j], Y[i, j])
    
    def log_levels(Z, num): 
        Z_min = np.min(Z)
        Z_shifted = Z - Z_min + 1e-1
        num_levels = num
        log_levels = np.logspace(np.log10(1e-1), np.log10(np.max(Z_shifted)), num=num_levels)
        levels_Z = log_levels + Z_min
        return levels_Z
    
    contour1 = ax1.contour(X, Y, Z_orig, levels=log_levels(Z_orig, 50), colors='black', alpha=0.6)
    ax1.clabel(contour1, inline=True, fontsize=8)
    ax1.set_title('Original Muller-Brown Potential')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Biased potential surface
    X_bias, Y_bias, Z_bias = abc_sim.compute_free_energy_surface()
    contour2 = ax2.contour(X_bias, Y_bias, Z_bias, levels=log_levels(Z_orig, 50), colors='blue', alpha=0.6)
    ax2.clabel(contour2, inline=True, fontsize=8)
    
    # Overlay bias positions
    bias_centers = abc_sim.get_bias_centers()
    if len(bias_centers) > 0:
        ax2.scatter(bias_centers[:, 0], bias_centers[:, 1], 
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

    # Create a time-based colormap (normalized from 0 to 1)
    t = np.linspace(0, len(trajectory), len(segments))
    lc = LineCollection(segments, cmap='viridis', array=t, linewidth=2, alpha=0.8)

    ax3.add_collection(lc)

    # Plot start and end points
    ax3.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, 
                marker='o', label='Start', zorder=5)
    ax3.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, 
                marker='s', label='End', zorder=5)

    # Plot bias centers if present
    if len(bias_centers) > 0:
        ax3.scatter(bias_centers[:, 0], bias_centers[:, 1], 
                    c='red', s=10, marker='x', linewidths=2, label='Biases', alpha=0.8, zorder=4)
        
    # Set fixed axis bounds
    # ax3.set_xlim(-2, 2)
    # ax3.set_ylim(-1, 2)

    # Aesthetic stuff
    ax3.set_title('ABC Trajectory')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # ax3.set_aspect('equal')

    # Add colorbar for time
    cbar = plt.colorbar(lc, ax=ax3, orientation='vertical')
    cbar.set_label('Simulation Time Step')
    
    # Plot 4: Exploration metrics
    # Calculate variance over time to show exploration
    window_size = 500
    variances = []
    times = []
    
    for i in range(window_size, len(trajectory), 100):
        window = trajectory[i-window_size:i]
        var_x = np.var(window[:, 0])
        var_y = np.var(window[:, 1])
        total_var = var_x + var_y
        variances.append(total_var)
        times.append(i)
    
    ax4.plot(times, variances, 'b-', linewidth=2)
    ax4.set_title('Exploration Measure (Position Variance)')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Total Variance')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
        
    if save_plots:
        plt.savefig('abc_dynamic_deposition.png', dpi=300, bbox_inches='tight')

    plt.show()

def analyze_basin_visits(trajectory, basin_centers, basin_radius=0.5):
    """
    Analyze which basins were visited during the simulation.
    
    Parameters:
    -----------
    trajectory : ndarray
        Trajectory of the simulation
    basin_centers : list of tuples
        Known basin centers in the Muller-Brown potential
    basin_radius : float
        Radius to consider for basin occupancy
    """
    # Known approximate basin centers for Muller-Brown
    if basin_centers is None:
        basin_centers = [
            (-0.558, 1.442),  # Basin A
            (0.623, 0.028),   # Basin B  
            (-0.050, 0.467)   # Basin C
        ]
    
    basin_visits = {i: [] for i in range(len(basin_centers))}
    current_basin = None
    
    for step, pos in enumerate(trajectory):
        # Find which basin we're in
        in_basin = None
        for j, center in enumerate(basin_centers):
            distance = np.linalg.norm(pos - np.array(center))
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

def main():
    """Run standard ABC simulation and analysis."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Starting Standard ABC Simulation")
    print("=" * 50)
    
    # Initialize ABC with standard parameters
    abc = StandardABC(
        dt=0.001,
        gamma=1.0,
        T=0.1,
        bias_height=10.0,
        bias_sigma=0.3,
        deposition_frequency=100,
        basin_threshold=0.1, 
        starting_position=(0,0),
        force_threshold=100
    )
    
    # Run simulation
    abc.run_simulation(max_steps=20000, verbose=True)
    
    # Analyze results
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total steps: {len(trajectory)}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: ({abc.position[0]:.3f}, {abc.position[1]:.3f})")
    
    # Analyze basin exploration
    analyze_basin_visits(trajectory, None)
    
    # Plot results
    plot_results(abc, save_plots=True)
    
    return abc

if __name__ == "__main__":
    abc_simulation = main()