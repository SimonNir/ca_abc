import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import deque
import copy

###############################
# Potential Functions (1D)
###############################

def double_well(x):
    """Compute double well potential with minima at x=-1 and x=1."""
    return (x**2 - 1)**2

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
        """
        self.dt = dt
        self.gamma = gamma
        self.T = T
        self.bias_height = bias_height
        self.bias_sigma = bias_sigma
        self.deposition_frequency = deposition_frequency
        self.basin_threshold = basin_threshold
        
        # Noise parameters
        self.noise_std = np.sqrt(2 * T * dt / gamma)
        
        # State variables
        self.position = np.array(-1.0)  # Start at left minimum
        self.bias_list = []  # List of (center, sigma, height) tuples
        self.trajectory = []
        self.step_count = 0
        self.last_bias_position = None
        self.inferred_minima = []  # Track minima inferred by the agent
        
    def reset(self, start_pos=None):
        """Reset the simulation."""
        if start_pos is None:
            self.position = np.array(-1.0)
        else:
            self.position = np.array(start_pos)
        self.bias_list = []
        self.trajectory = [self.position.copy()]
        self.step_count = 0
        self.last_bias_position = None
        self.inferred_minima = []
        
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
        # self.update_inferred_minima()
        
    def update_inferred_minima(self):
        """Update the inferred minima based on current bias positions."""
        if len(self.bias_list) < 2:
            return
            
        # Get all bias centers sorted
        centers = sorted([b[0] for b in self.bias_list])
        
        # The minima are the midpoints between bias centers
        new_minima = []
        for i in range(len(centers)-1):
            new_minima.append((centers[i] + centers[i+1])/2)
        
        self.inferred_minima = new_minima
        
    def langevin_step(self):
        """Perform one step of Langevin dynamics."""
        force = compute_force_1d(self.position, self.bias_list)
        noise = np.random.normal(0, self.noise_std)
        self.position += (force / self.gamma) * self.dt + noise
        self.position = np.clip(self.position, -2, 2)
        
    def run_simulation(self, max_steps=10000, verbose=True):
        """Run the ABC simulation."""
        for step in range(max_steps):
            self.step_count = step
            if self.should_deposit_bias():
                self.deposit_bias()
            self.langevin_step()
            self.trajectory.append(self.position.copy())
            if verbose and step % 1000 == 0:
                print(f"Step {step}: Position {self.position:.3f}, "
                      f"Biases: {len(self.bias_list)}")
                      
        print(f"Simulation completed. Total biases: {len(self.bias_list)}")
        self.update_inferred_minima()
        print(f"Inferred minima (after simulation): {self.inferred_minima}")
        
    def get_trajectory(self):
        return np.array(self.trajectory)
        
    def get_bias_centers(self):
        return np.array([b[0] for b in self.bias_list]) if self.bias_list else np.array([])
        
    def compute_free_energy_surface(self, x_range=(-2, 2), resolution=100):
        """Compute the free energy surface including biases."""
        x = np.linspace(x_range[0], x_range[1], resolution)
        F = np.zeros_like(x)
        for i in range(resolution):
            V = double_well(x[i])
            for center, sigma, height in self.bias_list:
                V += gaussian_bias_1d(x[i], center, sigma, height)
            F[i] = V
        return x, F

###############################
# Analysis and Visualization
###############################

def plot_results_1d(abc_sim, save_plots=False):
    """Plot the results with true and inferred minima."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Potential landscape
    x, F_orig = abc_sim.compute_free_energy_surface()
    ax1.plot(x, double_well(x), 'k-', label='True Potential')
    ax1.plot(x, F_orig, 'b-', alpha=0.7, label='Biased Potential')
    
    # Mark true minima
    ax1.axvline(-1, color='g', linestyle=':', alpha=0.7, label='True Minima')
    ax1.axvline(1, color='g', linestyle=':', alpha=0.7)
    ax1.text(-1, -0.5, 'True Min', ha='center', color='g')
    ax1.text(1, -0.5, 'True Min', ha='center', color='g')
    
    # Mark inferred minima
    if abc_sim.inferred_minima:
        for i, min_pos in enumerate(abc_sim.inferred_minima):
            ax1.axvline(min_pos, color='r', linestyle='--', alpha=0.7, 
                        label='Inferred Minima' if i==0 else None)
            ax1.text(min_pos, 0.5, f'Inferred Min {i+1}', 
                    ha='center', color='r')
    
    # Mark bias centers
    bias_centers = abc_sim.get_bias_centers()
    for center in bias_centers:
        ax1.axvline(center, color='orange', linestyle='-.', alpha=0.5, 
                   label='Bias Positions' if center==bias_centers[0] else None)
    
    ax1.set_title('Potential Landscape (True vs Inferred Minima)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Energy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Trajectory with transitions
    trajectory = abc_sim.get_trajectory()
    times = np.arange(len(trajectory))
    
    # Highlight transitions
    in_min1 = np.abs(trajectory + 1) < 0.3
    in_min2 = np.abs(trajectory - 1) < 0.3
    transitions = ~(in_min1 | in_min2)
    
    ax2.plot(times, trajectory, 'g-', alpha=0.3, linewidth=1)

    # Get segments where transitions change
    change_points = np.where(np.diff(transitions.astype(int)) != 0)[0] + 1
    for seg in np.split(np.arange(len(times)), change_points):
        if len(seg) > 1 and transitions[seg[0]]:
            ax2.plot(times[seg], trajectory[seg], 'r-', linewidth=2, alpha=0.7)

    
    # Mark minima
    ax2.axhline(-1, color='g', linestyle=':', alpha=0.7, label='True Minima')
    ax2.axhline(1, color='g', linestyle=':', alpha=0.7)
    if abc_sim.inferred_minima:
        for min_pos in abc_sim.inferred_minima:
            ax2.axhline(min_pos, color='r', linestyle='--', alpha=0.3, 
                       label='Inferred Minima' if min_pos==abc_sim.inferred_minima[0] else None)
    
    ax2.set_title('Trajectory (Red = Transition Paths)')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Histogram with inferred minima
    ax3.hist(trajectory, bins=50, density=True, color='purple', alpha=0.7)
    ax3.axvline(-1, color='g', linestyle=':', alpha=0.7, label='True Minima')
    ax3.axvline(1, color='g', linestyle=':', alpha=0.7)
    if abc_sim.inferred_minima:
        for min_pos in abc_sim.inferred_minima:
            ax3.axvline(min_pos, color='r', linestyle='--', alpha=0.7, 
                       label='Inferred Minima' if min_pos==abc_sim.inferred_minima[0] else None)
    
    ax3.set_title('Position Distribution')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('abc_1d_inferred_minima.png', dpi=300)
    plt.show()

def analyze_basin_visits_1d(trajectory, abc_sim):
    """Analyze visits to both true and inferred basins."""
    true_minima = [-1.0, 1.0]
    inferred_minima = abc_sim.inferred_minima if abc_sim.inferred_minima else []
    
    print("\nBasin Analysis:")
    print("-"*40)
    print("True Minima:")
    for i, center in enumerate(true_minima):
        in_basin = np.sum(np.abs(trajectory - center) < 0.3)
        print(f"  Basin {i} (x={center:.2f}): {in_basin} visits")
    
    if inferred_minima:
        print("\nInferred Minima:")
        for i, center in enumerate(inferred_minima):
            in_basin = np.sum(np.abs(trajectory - center) < 0.3)
            print(f"  Basin {i} (x={center:.2f}): {in_basin} visits")
    else:
        print("\nNo inferred minima detected")

###############################
# Main Execution
###############################

def main_1d():
    np.random.seed(42)
    print("Starting 1D ABC Simulation")
    
    abc = StandardABC1D(
        dt=0.001,
        gamma=1.0,
        T=0.5,
        bias_height=0.2,
        bias_sigma=0.1,
        deposition_frequency=100,
        basin_threshold=0.1
    )
    
    abc.run_simulation(max_steps=10000)
    
    trajectory = abc.get_trajectory()
    print(f"\nFinal position: {abc.position:.3f}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    if abc.inferred_minima:
        print(f"Inferred minima at: {abc.inferred_minima}")
    else:
        print("No minima inferred")
    
    analyze_basin_visits_1d(trajectory, abc)
    plot_results_1d(abc, save_plots=True)
    
    return abc

if __name__ == "__main__":
    abc_simulation_1d = main_1d()