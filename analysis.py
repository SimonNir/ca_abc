"""
Enhanced Analysis Module for ABC Simulations

Combines visualization and diagnostic tools for analyzing ABC simulation results.
Includes both standard plotting functions and advanced debugging metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.distance import cdist

class ABCAnalysis:
    """
    Comprehensive analysis toolkit for ABC simulations.
    
    Features:
    - Standard visualization (potential landscapes, trajectories)
    - Basin transition analysis
    - Force and energy diagnostics
    - Perturbation vs optimization step tracking
    - Curvature and barrier height estimation
    - Exploration metrics
    """
    
    def __init__(self, abc_sim):
        """
        Initialize with an ABC simulation object.
        
        Args:
            abc_sim: ABC simulation instance (TraditionalABC or SmartABC)
        """
        self.abc = abc_sim
        self.dimension = abc_sim.dimension
        
    def plot_summary(self, filename=None, save_plots=False):
        """
        Generate standard summary plots appropriate for system dimension.
        """
        if self.dimension == 1:
            self._plot_1d_summary(filename, save_plots)
        elif self.dimension == 2:
            self._plot_2d_summary(filename, save_plots)
        else:
            print(f"Standard visualization not supported for {self.dimension}D systems")
            self.plot_diagnostics()
    
    def _plot_1d_summary(self, filename=None, save_plots=False):
        """1D summary plots (potential, trajectory, histogram, energy profile)."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Potential landscape
        x, F_orig = self.abc.compute_free_energy_surface(resolution=1000)
        ax1.plot(x, self.abc.potential.potential(x), 'k-', label='Original Potential')
        ax1.plot(x, F_orig, 'b-', alpha=0.7, label='Biased Potential')
        
        # Mark bias centers
        bias_centers = self.abc.get_bias_centers()
        if len(bias_centers) > 0:
            for center in bias_centers:
                ax1.axvline(center[0], color='r', linestyle='--', alpha=0.5)
        
        ax1.set_title('Potential Energy Landscape')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Energy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trajectory with perturbation markers
        trajectory = self.abc.get_trajectory()
        times = np.arange(len(trajectory))
        ax2.plot(times, trajectory, 'g-', alpha=0.7, linewidth=1)
        
        # Mark perturbation steps - convert to integers first
        pert_steps = self._get_perturbation_steps()
        if pert_steps.size > 0:
            ax2.scatter(pert_steps, trajectory[pert_steps], 
                    c='red', s=20, marker='o', label='Perturbations')
        
        # Mark known basins
        if hasattr(self.abc.potential, 'known_minima'):
            for basin in self.abc.potential.known_minima():
                ax2.axhline(basin[0], color='k', linestyle=':', alpha=0.3)
        
        ax2.set_title('ABC Trajectory')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Position')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Position histogram
        ax3.hist(trajectory, bins=50, density=True, color='purple', alpha=0.7)
        ax3.set_title('Visited Positions Distribution')
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        if hasattr(self.abc.potential, 'plot_range'):
            ax3.set_xlim(self.abc.potential.plot_range())
        
        # Plot 4: Energy diagnostics
        self._plot_energy_diagnostics(ax4)
        
        plt.tight_layout()
        self._save_plot(fig, filename, save_plots)
        plt.show()
    
    def _plot_2d_summary(self, filename=None, save_plots=False):
        """2D summary plots (potential surfaces, trajectory, exploration)."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Original potential
        (X, Y), _ = self.abc.compute_free_energy_surface()
        Z_orig = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_orig[i,j] = self.abc.potential.potential(np.array([X[i,j], Y[i,j]]))
        
        contour1 = ax1.contour(X, Y, Z_orig, levels=self._log_levels(Z_orig, 50), 
                              colors='black', alpha=0.6)
        ax1.clabel(contour1, inline=True, fontsize=8)
        ax1.set_title('Original Potential')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Biased potential with bias centers
        (X_bias, Y_bias), Z_bias = self.abc.compute_free_energy_surface()
        contour2 = ax2.contour(X_bias, Y_bias, Z_bias, levels=self._log_levels(Z_orig, 50), 
                              colors='blue', alpha=0.6)
        ax2.clabel(contour2, inline=True, fontsize=8)
        
        bias_centers = self.abc.get_bias_centers()
        if len(bias_centers) > 0:
            ax2.scatter(bias_centers[:,0], bias_centers[:,1], 
                       c='red', s=50, marker='x', linewidths=2, label='Bias Centers')
            ax2.legend()
        
        ax2.set_title('Biased Potential Surface')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trajectory with perturbations
        trajectory = self.abc.get_trajectory()
        if len(trajectory) > 1:
            points = np.array(trajectory).reshape(-1,1,2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Color by time and mark perturbations
            t = np.linspace(0, len(trajectory), len(segments))
            lc = LineCollection(segments, cmap='viridis', array=t, linewidth=2, alpha=0.8)
            ax3.add_collection(lc)
            
            pert_steps = self._get_perturbation_steps()
            if pert_steps.size > 0:
                pert_pos = np.array(trajectory)[pert_steps]
                ax3.scatter(pert_pos[:,0], pert_pos[:,1], c='red', s=50, 
                           marker='*', label='Perturbations', zorder=5)
            
            ax3.set_title('ABC Trajectory')
            ax3.legend()
            plt.colorbar(lc, ax=ax3, label='Simulation Time Step')
        else:
            ax3.text(0.5, 0.5, 'No trajectory data', ha='center', va='center')
        
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Exploration metrics
        self._plot_exploration_metrics(ax4)
        
        plt.tight_layout()
        self._save_plot(fig, filename, save_plots)
        plt.show()
    
    def plot_diagnostics(self, filename=None, save_plots=False):
        """
        Generate diagnostic plots for debugging and optimization.
        
        Includes:
        - Force magnitude over time
        - Energy changes
        - Step sizes
        - Barrier height estimates
        - Curvature information
        """
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        # Plot 1: Force magnitude
        forces = self.abc.get_forces()
        force_mags = np.linalg.norm(forces, axis=1) if forces.ndim > 1 else np.abs(forces)
        axes[0].plot(force_mags, 'b-')
        axes[0].set_title('Force Magnitude Over Time')
        axes[0].set_ylabel('|Force|')
        axes[0].grid(True, alpha=0.3)
        
        # Mark perturbation steps
        pert_steps = self._get_perturbation_steps()
        if pert_steps.size > 0:
            axes[0].scatter(pert_steps, force_mags[pert_steps], c='r', s=30, label='Perturbations')
            axes[0].legend()
        
        # Plot 2: Energy profile
        self._plot_energy_diagnostics(axes[1])
        
        # Plot 3: Step sizes
        trajectory = self.abc.get_trajectory()
        if len(trajectory) > 1:
            steps = np.diff(trajectory, axis=0)
            step_sizes = np.linalg.norm(steps, axis=1) if steps.ndim > 1 else np.abs(steps)
            axes[2].plot(step_sizes, 'g-')
            axes[2].set_title('Step Sizes')
            axes[2].set_ylabel('Distance')
            axes[2].grid(True, alpha=0.3)
            
            if pert_steps.size > 0:
                pert_sizes = step_sizes[np.clip(pert_steps, 0, len(step_sizes)-1)]
                axes[2].scatter(pert_steps, pert_sizes, c='r', s=30)
        
        # Plot 4: Barrier height estimates
        if hasattr(self.abc, 'energies') and len(self.abc.energies) > 0:
            energies = np.array(self.abc.energies)
            if energies.ndim > 1:
                energies = energies.flatten()
            
            # Simple barrier estimation between minima
            minima = self._find_local_minima(energies)
            if len(minima) > 1:
                barriers = []
                for i in range(len(minima)-1):
                    start, end = minima[i], minima[i+1]
                    segment = energies[start:end]
                    if len(segment) > 0:
                        barrier = np.max(segment) - (energies[start] + energies[end])/2
                        barriers.append(barrier)
                
                axes[3].bar(range(len(barriers)), barriers)
                axes[3].set_title('Estimated Barrier Heights')
                axes[3].set_ylabel('Energy')
                axes[3].grid(True, alpha=0.3)
        
        # Plot 5: Curvature information if available
        if hasattr(self.abc, 'most_recent_hessian') and self.abc.most_recent_hessian is not None:
            eigvals = np.linalg.eigvalsh(self.abc.most_recent_hessian)
            axes[4].bar(range(len(eigvals)), eigvals)
            axes[4].set_title('Hessian Eigenvalues')
            axes[4].set_ylabel('Curvature')
            axes[4].grid(True, alpha=0.3)
        
        # Plot 6: Exploration metrics
        self._plot_exploration_metrics(axes[5])
        
        plt.tight_layout()
        self._save_plot(fig, filename, save_plots)
        plt.show()
    
    def _plot_energy_diagnostics(self, ax):
        """Plot energy diagnostics on provided axis."""
        if hasattr(self.abc, 'energies') and len(self.abc.energies) > 0:
            energies = np.array(self.abc.energies)
            if energies.ndim > 1:
                energies = energies.flatten()
            
            ax.plot(energies, 'orange', label='Unbiased PES')
            
            # Plot biased energies if we can compute them
            try:
                biased_energies = [self.abc.total_potential(pos) for pos in self.abc.trajectory]
                ax.plot(biased_energies, 'blue', alpha=0.5, label='Biased PES')
            except:
                pass
            
            ax.set_title('Energy Profile')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Energy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Mark perturbation steps
            pert_steps = self._get_perturbation_steps()
            if pert_steps.size > 0:
                ax.scatter(pert_steps, energies[pert_steps], c='r', s=20)
    
    def _plot_exploration_metrics(self, ax):
        """Plot exploration metrics on provided axis."""
        trajectory = self.abc.get_trajectory()
        if len(trajectory) > 2:
            # Cumulative distance from start
            start = trajectory[0]
            distances = cdist(trajectory, [start]).flatten()
            
            # Rolling variance
            window_size = min(10, len(trajectory)//3)
            variances = []
            for i in range(len(trajectory)-window_size):
                window = trajectory[i:i+window_size]
                variances.append(np.sum(np.var(window, axis=0)))
            
            ax.plot(distances, 'b-', label='Distance from Start')
            ax.plot(np.arange(window_size, len(trajectory)), variances, 
                   'g-', label=f'Rolling Variance (window={window_size})')
            ax.set_title('Exploration Metrics')
            ax.set_xlabel('Time Step')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
    
    def analyze_basin_visits(self, basin_radius=0.3, verbose=True):
        """
        Analyze which basins were visited during the simulation.
        
        Returns:
            Dictionary mapping basin indices to visit times
        """
        trajectory = self.abc.get_trajectory()
        
        if not hasattr(self.abc.potential, 'known_minima'):
            if verbose:
                print("No known basins defined for this potential")
            return {}
            
        known_basins = [np.atleast_1d(b) for b in self.abc.potential.known_minima()]
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
    
    def transition_statistics(self, iteration=None):
        """
        Compute statistics for a specific iteration or all iterations.
        
        Returns:
            Dictionary containing:
            - positions
            - unbiased energies
            - biased energies
            - forces
            - step types (perturbation/descent)
        """
        if iteration is not None:
            # Get data for specific iteration
            start_idx = np.sum(self.abc.iter_periods[:iteration])
            end_idx = np.sum(self.abc.iter_periods[:iteration+1])
            times = np.arange(start_idx, end_idx)
            traj = self.abc.trajectory[start_idx:end_idx]
            pes = self.abc.energies[start_idx:end_idx]
            forces = self.abc.forces[start_idx:end_idx]
            
            # Calculate biased PES
            biased_pes = pes.copy()
            for bias in self.abc.bias_list[:iteration]:  # Only use biases up to current iteration
                biased_pes += np.array([bias.potential(p) for p in traj])
            
            return {
                'positions': traj,
                'unbiased_energies': pes,
                'biased_energies': biased_pes,
                'forces': forces,
                'times': times
            }
        else:
            # Return statistics for all iterations
            results = []
            for i in range(len(self.abc.iter_periods)):
                results.append(self.transition_statistics(i))
            return results
    
    def _get_perturbation_steps(self):
        """Identify steps where perturbations occurred."""
        if not hasattr(self.abc, 'iter_periods'):
            return []
        
        pert_steps = []
        tot_steps = 0
        for period in self.abc.iter_periods:
            # Last step of each period is the perturbation
            if tot_steps + period - 1 < len(self.abc.trajectory):
                pert_steps.append(tot_steps + period - 1)
            tot_steps += period
        return np.array(pert_steps, dtype=int)
    
    def _find_local_minima(self, values, window=5):
        """Find indices of local minima in a 1D array."""
        minima = []
        for i in range(window, len(values)-window):
            if (values[i] <= values[i-window:i+window]).all():
                minima.append(i)
        return minima
    
    def _log_levels(self, Z, num):
        """Generate log-spaced contour levels."""
        Z_min = np.min(Z)
        Z_shifted = Z - Z_min + 1e-1
        log_levels = np.logspace(np.log10(1e-1), np.log10(np.max(Z_shifted)), num=num)
        return log_levels + Z_min
    
    def _save_plot(self, fig, filename, save_plots):
        """Save plot if requested."""
        if save_plots:
            if filename is None:
                filename = "abc_results.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')


# Legacy functions for backward compatibility
def plot_results(abc_sim, filename=None, save_plots=False):
    """Legacy function - use ABCAnalysis class for new code."""
    analyzer = ABCAnalysis(abc_sim)
    analyzer.plot_summary(filename, save_plots)

def analyze_basin_visits(abc_sim, basin_radius=0.3, verbose=True):
    """Legacy function - use ABCAnalysis class for new code."""
    analyzer = ABCAnalysis(abc_sim)
    return analyzer.analyze_basin_visits(basin_radius, verbose)