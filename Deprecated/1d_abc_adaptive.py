import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import defaultdict

###############################
# Potential Functions (1D)
###############################

def multi_min_potential(x):
    """
    Complex 1D potential with multiple minima:
    Combination of polynomial and trigonometric terms
    Minima will be automatically detected
    """
    return (x**4 - 3*x**2 + 0.5*x + 
            0.5*np.sin(5*x) + 
            0.3*np.cos(10*x) + 
            0.2*np.exp(-(x-1.5)**2/(2*0.2**2)))

def find_potential_minima(potential_func, x_range=(-2.5, 2.5), resolution=1000):
    """Automatically find minima locations in the potential"""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = potential_func(x)
    
    # Find minima by looking for peaks in the inverted potential
    minima_indices, _ = find_peaks(-y, prominence=0.1)
    minima = x[minima_indices]
    
    # Filter very shallow minima
    significant_minima = []
    for m in minima:
        # Check if this is really a minimum by checking nearby points
        if (potential_func(m-0.05) > potential_func(m) and 
            potential_func(m+0.05) > potential_func(m)):
            significant_minima.append(m)
    
    return np.array(significant_minima)

def gaussian_bias_1d(x, center, sigma, height):
    """Compute 1D Gaussian bias potential."""
    dx = x - center
    exponent = -dx**2 / (2 * sigma**2)
    exponent = np.clip(exponent, -100, 100)
    return height * np.exp(exponent)

def compute_force_1d(x, bias_list, eps=1e-5):
    """Compute negative gradient of total potential (force) in 1D."""
    def total_potential(pos_x):
        V = multi_min_potential(pos_x)
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
# ABC Implementation with Automatic Minima Detection
###############################

class AdaptiveABC1D:
    def __init__(self, dt=0.01, gamma=1.0, T=0.1, bias_height=10.0, 
                 bias_sigma=0.2, deposition_frequency=100, 
                 basin_threshold=0.15, convergence_window=1000,
                 min_basin_visits=3):
        """
        1D ABC with automatic minima detection and convergence.
        """
        self.dt = dt
        self.gamma = gamma
        self.T = T
        self.bias_height = bias_height
        self.bias_sigma = bias_sigma
        self.deposition_frequency = deposition_frequency
        self.basin_threshold = basin_threshold
        self.convergence_window = convergence_window
        self.min_basin_visits = min_basin_visits
        
        # Noise parameters
        self.noise_std = np.sqrt(2 * T * dt / gamma)
        
        # State variables
        self.position = np.array(0.0)
        self.bias_list = []
        self.trajectory = []
        self.step_count = 0
        self.last_bias_position = None
        self.converged = False
        
        # Minima detection
        self.minima = find_potential_minima(multi_min_potential)
        print(f"Detected potential minima at: {self.minima.round(2)}")
        
        # For convergence detection
        self.basin_history = []
        self.unique_basins = set()
        self.convergence_counter = 0
        
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
        self.converged = False
        self.basin_history = []
        self.unique_basins = set()
        self.convergence_counter = 0
        
    def detect_current_basin(self):
        """Determine which basin the particle is currently in."""
        if len(self.minima) == 0:
            return None
            
        distances = np.abs(self.position - self.minima)
        closest_idx = np.argmin(distances)
        
        if distances[closest_idx] < self.basin_threshold:
            return closest_idx
        return None
        
    def should_deposit_bias(self):
        """Determine if a bias should be deposited."""
        if self.step_count % self.deposition_frequency != 0:
            return False
            
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
        force = compute_force_1d(self.position, self.bias_list)
        noise = np.random.normal(0, self.noise_std)
        self.position += (force / self.gamma) * self.dt + noise
        self.position = np.clip(self.position, -2.5, 2.5)
        
    def check_convergence(self):
        """Check if we've found all basins and can stop."""
        current_basin = self.detect_current_basin()
        self.basin_history.append(current_basin)
        
        # Only check periodically
        if self.step_count % 100 != 0:
            return False
        
        # Count basin visits in recent history
        recent_history = self.basin_history[-self.convergence_window:]
        basin_counts = defaultdict(int)
        for basin in recent_history:
            if basin is not None:
                basin_counts[basin] += 1
        
        # Check if we've sufficiently visited all basins
        all_basins_found = True
        for basin_idx in range(len(self.minima)):
            if basin_counts.get(basin_idx, 0) < self.min_basin_visits:
                all_basins_found = False
                break
        
        if all_basins_found and len(basin_counts) >= len(self.minima):
            self.convergence_counter += 1
            if self.convergence_counter >= 5:  # Require consistent convergence
                print(f"\nConvergence reached at step {self.step_count}")
                print(f"Found all {len(self.minima)} basins")
                self.converged = True
                return True
        else:
            self.convergence_counter = 0
            
        return False
        
    def run_simulation(self, max_steps=30000, verbose=True):
        """Run the ABC simulation with convergence checking."""
        for step in range(max_steps):
            self.step_count = step
            
            if self.should_deposit_bias():
                self.deposit_bias()
                
            self.langevin_step()
            self.trajectory.append(self.position.copy())
            
            if self.check_convergence():
                break
                
            if verbose and step % 1000 == 0:
                current_basin = self.detect_current_basin()
                basin_info = f" (Basin {current_basin})" if current_basin is not None else ""
                print(f"Step {step}: Position {self.position:.3f}{basin_info}, "
                      f"Biases: {len(self.bias_list)}")
                      
        print(f"\nSimulation completed after {self.step_count} steps")
        print(f"Total biases deposited: {len(self.bias_list)}")
        
    def get_trajectory(self):
        return np.array(self.trajectory)
        
    def get_bias_centers(self):
        if not self.bias_list:
            return np.array([])
        return np.array([bias[0] for bias in self.bias_list])
        
    def compute_free_energy_surface(self, x_range=(-2.5, 2.5), resolution=200):
        x = np.linspace(x_range[0], x_range[1], resolution)
        F = np.zeros_like(x)
        for i in range(resolution):
            V = multi_min_potential(x[i])
            for center, sigma, height in self.bias_list:
                V += gaussian_bias_1d(x[i], center, sigma, height)
            F[i] = V
        return x, F

###############################
# Visualization
###############################

def plot_results_1d(abc_sim, save_plots=False):
    """Enhanced visualization with proper minima labeling."""
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2)
    
    # Plot 1: Potential landscape
    ax1 = fig.add_subplot(gs[0, :])
    x, F_orig = abc_sim.compute_free_energy_surface()
    ax1.plot(x, multi_min_potential(x), 'k-', label='Original Potential')
    ax1.plot(x, F_orig, 'b-', alpha=0.7, label='Biased Potential')
    
    # Mark bias centers
    bias_centers = abc_sim.get_bias_centers()
    if len(bias_centers) > 0:
        for center in bias_centers:
            ax1.axvline(center, color='r', linestyle='--', alpha=0.3)
    
    # Mark detected minima
    minima = abc_sim.minima
    for i, m in enumerate(minima):
        ax1.axvline(m, color='g', linestyle=':', alpha=0.5)
        ax1.text(m, ax1.get_ylim()[0]+0.1, f'Min {i}', 
                ha='center', va='bottom', color='g')
    
    ax1.set_title('Potential Energy Landscape with Detected Minima')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Energy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Trajectory
    ax2 = fig.add_subplot(gs[1, 0])
    trajectory = abc_sim.get_trajectory()
    times = np.arange(len(trajectory))
    ax2.plot(times, trajectory, 'g-', alpha=0.7, linewidth=0.5)
    
    # Mark minima
    for i, m in enumerate(minima):
        ax2.axhline(m, color='g', linestyle=':', alpha=0.3)
        ax2.text(0, m, f'Min {i}', ha='right', va='center', color='g')
    
    ax2.set_title('Trajectory')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Position histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(trajectory, bins=50, density=True, 
             color='purple', alpha=0.7, orientation='horizontal')
    for i, m in enumerate(minima):
        ax3.axhline(m, color='g', linestyle=':', alpha=0.3)
    ax3.set_title('Position Distribution')
    ax3.set_xlabel('Frequency')
    ax3.set_ylabel('Position')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Basin discovery
    ax4 = fig.add_subplot(gs[2, :])
    basin_history = []
    for pos in trajectory:
        found = None
        for i, m in enumerate(minima):
            if np.abs(pos - m) < abc_sim.basin_threshold:
                found = i
                break
        basin_history.append(found)
    
    # Smooth for visualization
    window_size = 100
    smoothed = np.convolve(
        [0 if x is None else x+1 for x in basin_history],
        np.ones(window_size)/window_size, mode='valid')
    
    ax4.plot(smoothed, 'b-', alpha=0.7)
    ax4.set_yticks(range(len(minima)+1))
    ax4.set_yticklabels(['None'] + [f'Min {i}' for i in range(len(minima))])
    ax4.set_title('Basin Exploration Over Time')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Current Basin')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('abc_1d_automatic_results.png', dpi=300)
    plt.show()

###############################
# Main Execution
###############################

def main_adaptive():
    """Run adaptive ABC simulation with automatic minima detection."""
    np.random.seed(42)
    
    print("Starting Adaptive 1D ABC Simulation with Automatic Minima Detection")
    print("=" * 60)
    
    abc = AdaptiveABC1D(
        dt=0.001,
        gamma=1.0,
        T=0.1,
        bias_height=15.0,
        bias_sigma=0.2,
        deposition_frequency=1000,
        basin_threshold=0.15,
        convergence_window=20000,
        min_basin_visits=30
    )
    
    abc.run_simulation(max_steps=50000, verbose=True)
    plot_results_1d(abc, save_plots=True)
    
    return abc

if __name__ == "__main__":
    abc_sim = main_adaptive()