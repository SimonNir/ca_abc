###############################
# Analysis and Visualization
###############################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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
    x, F_orig = abc_sim.compute_free_energy_surface(resolution=1000)
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
    
    # Mark known basins if available
    if hasattr(abc_sim.potential, 'known_basins'):
        for basin in abc_sim.potential.known_minima():
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
    if hasattr(abc_sim.potential, 'plot_range'):
        ax3.set_xlim(abc_sim.potential.plot_range())
    
    # Plot 4: Energy profile over time
    energy_profile = abc_sim.get_energies()
    # Ensure energy_profile is 1D array
    if energy_profile.ndim > 1:
        energy_profile = energy_profile.flatten()
    # Ensure times and energy_profile have same length
    if len(times) != len(energy_profile):
        min_len = min(len(times), len(energy_profile))
        times = times[:min_len]
        energy_profile = energy_profile[:min_len]
    
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
    trajectory = abc_sim.get_trajectory()
    if len(trajectory) > 1:
        points = np.array(trajectory).reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Time-based colormap
        t = np.linspace(0, len(trajectory), len(segments))
        lc = LineCollection(segments, cmap='viridis', array=t, linewidth=2, alpha=0.8)
        
        ax3.add_collection(lc)
        
        # Plot start and end points
        ax3.scatter(trajectory[0][0], trajectory[0][1], c='green', s=100, 
                    marker='o', label='Start', zorder=5)
        ax3.scatter(trajectory[-1][0], trajectory[-1][1], c='red', s=100, 
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
    else:
        ax3.text(0.5, 0.5, 'No trajectory data', ha='center', va='center')
        ax3.set_title('ABC Trajectory')
    
    # Plot 4: Exploration metrics
    if len(trajectory) > 2:
        window_size = 3
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
    else:
        ax4.text(0.5, 0.5, 'Not enough data for variance', ha='center', va='center')
        ax4.set_title('Exploration Measure')
    
    plt.tight_layout()
    
    if save_plots:
        if filename is None:
            filename = "2d_results.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def analyze_basin_visits(abc_sim, basin_radius=0.3, verbose=True):
    """
    Analyze which basins were visited during the simulation.
    """
    trajectory = abc_sim.get_trajectory()
    
    if not hasattr(abc_sim.potential, 'known_minima'):
        if verbose:
            print("No known basins defined for this potential")
        return {}
        
    known_basins = [np.atleast_1d(b) for b in abc_sim.potential.known_minima()]
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


def transition_pathway_stats(abc_sim, iter):
    """Plot the transition pathway for a given iteration.
    
    Args:
        abc_sim: TraditionalABC instance
        iter: iteration number to plot
    """
    # Get trajectory and energies for this iteration
    start_idx = np.sum(abc_sim.iter_periods[:iter])
    end_idx = np.sum(abc_sim.iter_periods[:iter+1])
    times = np.arange(start_idx, end_idx)
    traj = abc_sim.trajectory[times]
    pes = abc_sim.energies[times]
    
    # Calculate biased PES
    biased_pes = pes.copy()
    for bias in abc_sim.bias_list[:iter]:  # Only use biases up to current iteration
        biased_pes += bias(traj)
    
    # You might want to return the data for external plotting
    return {
        'positions': traj,
        'unbiased_energies': pes,
        'biased_energies': biased_pes,
        'times': times
    }



    """
    Let's make a list of all the plots we would want for a general n-d case
    Let's also consider these for debugging output (e.g. print the full new position vector and the distance)

    1. Biased and unbiased energy as a function of timestep and/or distance from starting point 

    2. Force magnitude and force calls as a function of timestep 

    3. Change in position compared to the prior call and/or the origin as a function of timestep

    4. Curvature (at minima) 

    5. Estimate of barrier height for a given iteration 

    6. In 2d and 1d, labeled plots showing the trajectories as function of step, with the perturbation counting as a step 

    7. Total energy and force calls 

    For all of these it would be nice to see where biases were placed and which steps were perturbations vs BFGS 

    This would allow us to tweak our hyperparameters 

    """