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
    - Minima and saddle point analysis
    - Force and energy diagnostics
    - Biases and perturbations tracking
    - Barrier height estimation
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
        
    def plot_summary(self, filename=None, save_plots=False, plot_type='both'):
        """
        Generate standard summary plots appropriate for system dimension.
        
        Args:
            plot_type: 'biases', 'perturbations', or 'both'
        """
        if self.dimension == 1:
            self._plot_1d_summary(filename, save_plots, plot_type)
        elif self.dimension == 2:
            self._plot_2d_summary(filename, save_plots, plot_type)
        else:
            print(f"Standard visualization not supported for {self.dimension}D systems")
            self.plot_diagnostics(plot_type=plot_type)
    
    def _plot_1d_summary(self, filename=None, save_plots=False, plot_type='both'):
        """1D summary plots (potential, trajectory, histogram, energy profile)."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Potential landscape
        x, F_orig = self.abc.compute_free_energy_surface(resolution=1000)
        ax1.plot(x, self.abc.potential.potential(x), 'k-', label='Original Potential')
        ax1.plot(x, F_orig, 'b-', alpha=0.7, label='Biased Potential')
        
        # Mark minima and saddles from ABCsim
        if hasattr(self.abc, 'minima'):
            for i, min_pos in enumerate(self.abc.minima):
                ax1.axvline(min_pos[0], color='g', linestyle='--', alpha=0.5, label='Minima' if i == 0 else None)
        
        if hasattr(self.abc, 'saddles'):
            for i, sad_pos in enumerate(self.abc.saddles):
                ax1.axvline(sad_pos[0], color='r', linestyle=':', alpha=0.5, label='Saddles' if i == 0 else None)
        
        ax1.set_title('Potential Energy Landscape')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Energy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trajectory with markers
        trajectory = self.abc.get_trajectory()
        times = np.arange(len(trajectory))
        ax2.plot(times, trajectory, 'g-', alpha=0.7, linewidth=1)
        
        # Mark biases and perturbations based on plot_type
        if plot_type in ['biases', 'both']:
            bias_steps = self._get_bias_steps()
            if bias_steps.size > 0:
                ax2.scatter(bias_steps, trajectory[bias_steps], 
                        c='red', s=20, marker='x', label='Biases')
        
        if plot_type in ['perturbations', 'both']:
            pert_steps = self._get_perturbation_steps()
            if pert_steps.size > 0:
                ax2.scatter(pert_steps, trajectory[pert_steps], 
                        c='blue', s=20, marker='o', label='Perturbations')
        
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
        
        # Plot 4: Energy and force diagnostics
        self._plot_exploration_metrics(ax4)
        
        plt.tight_layout()
        self._save_plot(fig, filename, save_plots)
        plt.show()
    
    def _plot_2d_summary(self, filename=None, save_plots=False, plot_type='both'):
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
        
        # Mark minima and saddles from ABCsim
        if hasattr(self.abc, 'minima'):
            ax1.scatter([m[0] for m in self.abc.minima], [m[1] for m in self.abc.minima],
                       c='g', marker='o', s=50, label='Minima')
        
        if hasattr(self.abc, 'saddles'):
            ax1.scatter([s[0] for s in self.abc.saddles], [s[1] for s in self.abc.saddles],
                       c='r', marker='x', s=50, label='Saddles')
        
        ax1.set_title('Original Potential')
        ax1.legend()
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
        
        # Plot 3: Trajectory with markers
        trajectory = self.abc.get_trajectory()
        if len(trajectory) > 1:
            points = np.array(trajectory).reshape(-1,1,2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Color by time
            t = np.linspace(0, len(trajectory), len(segments))
            lc = LineCollection(segments, cmap='viridis', array=t, linewidth=2, alpha=0.8)
            ax3.add_collection(lc)
            
            # Mark biases and perturbations based on plot_type
            if plot_type in ['biases', 'both']:
                bias_steps = self._get_bias_steps()
                if bias_steps.size > 0:
                    bias_pos = np.array(trajectory)[bias_steps]
                    ax3.scatter(bias_pos[:,0], bias_pos[:,1], c='blue', s=50, 
                               marker='o', label='Biases', zorder=5)
            
            if plot_type in ['perturbations', 'both']:
                pert_steps = self._get_perturbation_steps()
                if pert_steps.size > 0:
                    pert_pos = np.array(trajectory)[pert_steps]
                    ax3.scatter(pert_pos[:,0], pert_pos[:,1], c='red', s=50, 
                               marker='x', label='Perturbations', zorder=5)
            
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
    
    def plot_diagnostics(self, filename=None, save_plots=False, plot_type='both'):
        """
        Generate diagnostic plots for debugging and optimization.
        
        Includes:
        - Force magnitude over time (biased and unbiased)
        - Energy changes
        - Step sizes
        - Barrier height estimates between identified minima
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        tpw_info = self.transition_statistics()
        
        # Plot 1: Force magnitude (both biased and unbiased)
        forces = tpw_info['forces']
        force_mags = np.linalg.norm(forces, axis=1) if forces.ndim > 1 else np.abs(forces)
        axes[0].plot(force_mags, 'b-', label='Unbiased Force')
        
        biased_forces = tpw_info['biased_forces']
        biased_force_mags = np.linalg.norm(biased_forces, axis=1) if biased_forces.ndim > 1 else np.abs(biased_forces)
        axes[0].plot(biased_force_mags, 'c-', alpha=0.7, label='Biased Force')
        
        axes[0].set_title('Force Magnitude Over Time')
        axes[0].set_ylabel('|Force|')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        
        # Plot 2: Energy diagnostics
        energies = tpw_info['energies']
        biased_energies = tpw_info['biased_energies']
        axes[1].plot(energies, color='orange', label='Unbiased PES')
        axes[1].plot(biased_energies, color='blue', alpha=0.5, label='Biased PES')
        axes[1].set_title('Energy Profile')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Energy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Mark biases and perturbations based on plot_type
        if plot_type in ['biases', 'both']:
            bias_steps = self._get_bias_steps()
            if bias_steps.size > 0:
                axes[0].scatter(bias_steps, force_mags[bias_steps], c='red', s=30, label='Biases')
                axes[1].scatter(bias_steps, energies[bias_steps], c='red', s=30, label='Biases')                
        
        if plot_type in ['perturbations', 'both']:
            pert_steps = self._get_perturbation_steps()
            if pert_steps.size > 0:
                axes[0].scatter(pert_steps, force_mags[pert_steps], c='blue', s=30, label='Perturbations')
                axes[1].scatter(pert_steps, energies[pert_steps], c='blue', s=30, label='Perturbations')
        
        # Plot 3: Barrier height estimates between identified minima
        if hasattr(self.abc, 'energies') and len(self.abc.energies) > 0 and hasattr(self.abc, 'minima'):
            energies = np.array(self.abc.energies)
            if energies.ndim > 1:
                energies = energies.flatten()
            
            minima_indices = []
            for min_pos in self.abc.minima:
                # Find closest point in trajectory to each minimum
                traj = np.array(self.abc.get_trajectory())
                dists = np.linalg.norm(traj - min_pos, axis=1)
                minima_indices.append(np.argmin(dists))
            
            if len(minima_indices) > 1:
                barriers = []
                barrier_labels = []
                for i in range(len(minima_indices)-1):
                    start, end = minima_indices[i], minima_indices[i+1]
                    segment = energies[start:end]
                    if len(segment) > 0:
                        max_energy = np.max(segment)
                        barrier = max_energy - min(energies[start], energies[end])
                        barriers.append(barrier)
                        barrier_labels.append(f'{i}→{i+1}')
                
                if barriers:
                    axes[2].bar(barrier_labels, barriers)
                    axes[2].set_title('Barrier Heights Between Minima')
                    axes[2].set_ylabel('Energy')
                    axes[2].grid(True, alpha=0.3)
                else:
                    axes[2].text(0.5, 0.5, 'No barrier data', ha='center', va='center')
            else:
                axes[2].text(0.5, 0.5, 'Not enough minima', ha='center', va='center')
        else:
            axes[2].text(0.5, 0.5, 'No energy/minima data', ha='center', va='center')
        
        plt.tight_layout()
        self._save_plot(fig, filename, save_plots)
        plt.show()
    

    def _plot_exploration_metrics(self, ax):
        """Plot exploration metrics on provided axis."""
        trajectory = self.abc.get_trajectory()
        if len(trajectory) > 2:
            # Cumulative distance from start
            start = trajectory[0]
            distances = cdist(trajectory, [start]).flatten()
            
            # Rolling variance
            window_size = min(3, len(trajectory)//3)
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
    
    def analyze_minima_saddles(self, proximity_radius=None):
        """
        Analyze minima and saddles found during simulation.
        
        Args:
            proximity_radius: If not None, also counts points within this distance of minima/saddles
            
        Returns:
            Dictionary containing analysis of minima and saddles
        """
        analysis = {'minima': [], 'saddles': []}
        
        if not hasattr(self.abc, 'minima') or not self.abc.minima:
            print("No minima identified in simulation")
            return analysis
        
        trajectory = np.array(self.abc.get_trajectory())
        
        # Analyze minima
        print("\nMinima Analysis:")
        print("-" * 40)
        for i, min_pos in enumerate(self.abc.minima):
            min_data = {'position': min_pos}
            
            # Find closest point in trajectory
            dists = np.linalg.norm(trajectory - min_pos, axis=1)
            closest_idx = np.argmin(dists)
            min_data['closest_distance'] = dists[closest_idx]
            min_data['closest_energy'] = self.abc.energies[closest_idx] if hasattr(self.abc, 'energies') else None
            
            # Count points within proximity radius if requested
            if proximity_radius is not None:
                nearby = np.sum(dists < proximity_radius)
                min_data['points_within_radius'] = nearby
                print(f"Minimum {i} at {min_pos}:")
                print(f"  Closest approach: {min_data['closest_distance']:.3f} distance")
                print(f"  Closest energy: {min_data['closest_energy']:.3f}" if min_data['closest_energy'] is not None else "")
                print(f"  Points within {proximity_radius} radius: {nearby}")
            else:
                print(f"Minimum {i} at {min_pos}:")
                print(f"  Closest approach: {min_data['closest_distance']:.3f} distance")
                print(f"  Closest energy: {min_data['closest_energy']:.3f}" if min_data['closest_energy'] is not None else "")
            
            analysis['minima'].append(min_data)
        
        # Analyze saddles if available
        if hasattr(self.abc, 'saddles') and self.abc.saddles:
            print("\nSaddle Analysis:")
            print("-" * 40)
            for i, sad_pos in enumerate(self.abc.saddles):
                sad_data = {'position': sad_pos}
                
                # Find closest point in trajectory
                dists = np.linalg.norm(trajectory - sad_pos, axis=1)
                closest_idx = np.argmin(dists)
                sad_data['closest_distance'] = dists[closest_idx]
                sad_data['closest_energy'] = self.abc.energies[closest_idx] if hasattr(self.abc, 'energies') else None
                
                # Count points within proximity radius if requested
                if proximity_radius is not None:
                    nearby = np.sum(dists < proximity_radius)
                    sad_data['points_within_radius'] = nearby
                    print(f"Saddle {i} at {sad_pos}:")
                    print(f"  Closest approach: {sad_data['closest_distance']:.3f} distance")
                    print(f"  Closest energy: {sad_data['closest_energy']:.3f}" if sad_data['closest_energy'] is not None else "")
                    print(f"  Points within {proximity_radius} radius: {nearby}")
                else:
                    print(f"Saddle {i} at {sad_pos}:")
                    print(f"  Closest approach: {sad_data['closest_distance']:.3f} distance")
                    print(f"  Closest energy: {sad_data['closest_energy']:.3f}" if sad_data['closest_energy'] is not None else "")
                
                analysis['saddles'].append(sad_data)
        
        return analysis
    
    def _get_bias_steps(self):
        """Identify steps where biases were placed (step before perturbation)."""
        pert_steps = self._get_perturbation_steps()
        return pert_steps - 1 if pert_steps.size > 0 else np.array([], dtype=int)
    
    def _get_perturbation_steps(self):
        """Identify steps where perturbations occurred."""
        if not hasattr(self.abc, 'iter_periods'):
            return np.array([], dtype=int)
        
        pert_steps = []
        tot_steps = 0
        for period in self.abc.iter_periods:
            # Last step of each period is the perturbation
            if tot_steps + period - 1 < len(self.abc.trajectory):
                pert_steps.append(tot_steps + period - 1)
            tot_steps += period
        return np.array(pert_steps, dtype=int)
    
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
            biased_pes = self.abc.energies[start_idx:end_idx]
            biased_forces = self.abc.forces[start_idx:end_idx]
            
            # Calculate biased PES
            unbiased_pes = biased_pes.copy()
            unbiased_forces = biased_forces.copy()
            for bias in self.abc.bias_list[:iteration]:  # Only use biases up to current iteration
                unbiased_pes -= np.array([bias.potential(p) for p in traj])
                unbiased_forces -= np.array([-1*bias.gradient(p) for p in traj])
            
            return {
                'positions': traj,
                'unbiased_energies': unbiased_pes,
                'biased_energies': biased_pes,
                'unbiased_forces': unbiased_forces,
                'biased_forces': biased_forces,
                'times': times
            }
        else:
            # Return statistics for all iterations
            # Initialize containers for concatenated data
            all_traj = []
            all_pes = []
            all_biased_pes = []
            all_forces = []
            all_biased_forces = []
            all_times = []
            
            for i in range(len(self.abc.iter_periods)):
                # Get data for each iteration
                start_idx = int(np.sum(self.abc.iter_periods[:i]))
                end_idx = int(np.sum(self.abc.iter_periods[:i+1]))
                
                # Append data to containers
                all_traj.append(self.abc.trajectory[start_idx:end_idx])
                all_pes.append(self.abc.energies[start_idx:end_idx])
                all_times.append(np.arange(start_idx, end_idx))
                
                # Calculate forces and biased quantities
                forces = self.abc.forces[start_idx:end_idx]
                biased_pes = np.array(self.abc.energies[start_idx:end_idx].copy())
                biased_forces = np.array(forces.copy())
                
                for bias in self.abc.bias_list[:i]:
                    traj_slice = self.abc.trajectory[start_idx:end_idx]
                    # Ensure the bias potential outputs are the correct shape
                    bias_potentials = np.array([bias.potential(p) for p in traj_slice])
                    bias_gradients = np.array([bias.gradient(p) for p in traj_slice])
                    
                    # Reshape if necessary to match dimensions
                    if biased_pes.ndim == 2 and bias_potentials.ndim == 1:
                        bias_potentials = bias_potentials.reshape(-1, 1)
                    if biased_forces.ndim == 2 and bias_gradients.ndim == 1:
                        bias_gradients = bias_gradients.reshape(-1, 1)
                        
                    biased_pes += bias_potentials
                    biased_forces += -1*bias_gradients
                
                all_forces.append(forces)
                all_biased_pes.append(biased_pes)
                all_biased_forces.append(biased_forces)
            
            # Concatenate all arrays
            return {
                'positions': np.concatenate(all_traj),
                'energies': np.concatenate(all_pes),
                'biased_energies': np.concatenate(all_biased_pes),
                'forces': np.concatenate(all_forces),
                'biased_forces': np.concatenate(all_biased_forces),
                'times': np.concatenate(all_times)
            }


# Legacy functions for backward compatibility
def plot_results(abc_sim, filename=None, save_plots=False):
    """Legacy function - use ABCAnalysis class for new code."""
    analyzer = ABCAnalysis(abc_sim)
    analyzer.plot_summary(filename, save_plots)

def analyze_basin_visits(abc_sim, basin_radius=0.3, verbose=True):
    """Legacy function - use ABCAnalysis class for new code."""
    analyzer = ABCAnalysis(abc_sim)
    return analyzer.analyze_minima_saddles(basin_radius)