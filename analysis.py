import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.distance import cdist

class ABCAnalysis:
    """
    Simplified analysis toolkit for ABC class that works directly with simulation objects.
    All data is accessed directly from the ABC instance.
    """

    def __init__(self, abc):
        """
        Initialize with ABC simulation object.
        
        Args:
            abc: ABC simulation instance (TraditionalABC or SmartABC)
        """
        self.abc = abc
        self.dimension = abc.dimension

    # Data access methods
    def get_trajectory(self):
        """Get complete trajectory"""
        return np.array(self.abc.trajectory)

    def get_energies(self, biased=False):
        """Get energy values"""
        return np.array(self.abc.biased_energies if biased else self.abc.unbiased_energies)

    def get_forces(self, biased=False):
        """Get force values"""
        return np.array(self.abc.biased_forces if biased else self.abc.unbiased_forces)

    def get_bias_steps(self):
        """Identify steps where biases were placed"""
        periods = np.array(self.abc.iter_periods)
        
        def get_end_indices(lengths):
            current_index = -1
            end_indices = []
            for length in lengths:
                current_index += length
                end_indices.append(current_index)
            return np.array(end_indices, dtype=int)

        return get_end_indices(periods)

    def get_perturbation_steps(self):
        """Identify steps where perturbations were enacted"""
        bias_steps = self.get_bias_steps()
        if len(bias_steps) < 2:
            return np.array([], dtype=int)
        return bias_steps[:-1]+1  # 1 fewer perturb step than bias steps

    def compute_free_energy_surface(self, resolution=100):
        """
        Compute free energy surface on grid for visualization.
        
        Returns:
            - For 1D: (x, F) where x are positions and F are free energies
            - For 2D: ((X, Y), F) where X,Y are meshgrid and F are free energies
        """
        if self.dimension == 1:
            x_range = self.abc.potential.plot_range()
            x = np.linspace(x_range[0], x_range[1], resolution)
            F = np.array([self.abc.compute_biased_potential(np.array([xi])) for xi in x])
            return x, F
        elif self.dimension == 2:
            x_range, y_range = self.abc.potential.plot_range()
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x, y)
            F = np.zeros_like(X)
            for i in range(resolution):
                for j in range(resolution):
                    pos = np.array([X[i,j], Y[i,j]])
                    F[i,j] = self.abc.compute_biased_potential(pos)
            return (X, Y), F
        else:
            raise NotImplementedError("Visualization not implemented for dimensions > 2")

    # Plotting utilities
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

    def plot_summary(self, filename=None, save_plots=False, plot_type='neither'):
        """Generate standard summary plots appropriate for system dimension"""
        if self.dimension == 1:
            self._plot_1d_summary(filename, save_plots, plot_type)
        elif self.dimension == 2:
            self._plot_2d_summary(filename, save_plots, plot_type)
        else:
            print(f"Standard visualization not supported for {self.dimension}D systems")
            self.plot_diagnostics(plot_type=plot_type)

    def _plot_1d_summary(self, filename=None, save_plots=False, plot_type='neither'):
        """1D summary plots (potential, trajectory, histogram, energy profile)"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Potential landscape
        x, F_orig = self.compute_free_energy_surface(resolution=1000)
        ax1.plot(x, self.abc.potential.potential(x), 'k-', label='Original Potential')
        ax1.plot(x, F_orig, 'b-', alpha=0.7, label='Biased Potential')
        
        if plot_type in ["both", "biases"]:
            # Mark biases in red
            for i, bias in enumerate(self.abc.bias_list):
                ax1.axvline(bias.center[0], color='red', linestyle='-', alpha=0.1, label='Biases' if i == 0 else None)
        
        ax1.set_title('Potential Energy Landscape')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Energy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trajectory with markers (swap x and y axes)
        trajectory = self.get_trajectory()
        times = np.arange(len(trajectory))
        ax2.scatter(trajectory, times, color='g', marker='o', alpha=0.7, s=4, linewidth=1)
        
        # Mark biases and perturbations
        bias_steps = self.get_bias_steps()
        if bias_steps.size > 0 and plot_type in ['biases', 'both']:
            ax2.scatter(trajectory[bias_steps], bias_steps, c='red', s=4, marker='x', label='Biases')
        
        pert_steps = self.get_perturbation_steps()
        if pert_steps.size > 0 and plot_type in ['perturbations', 'both']:
            ax2.scatter(trajectory[pert_steps], pert_steps, c='blue', s=4, marker='o', label='Perturbations')
        
        ax2.set_title('ABC Trajectory')
        ax2.set_ylabel('Time Step')
        ax2.set_xlabel('Position')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Position histogram
        ax3.hist(trajectory, bins=50, density=True, color='purple', alpha=0.7)
        ax3.set_title('Visited Positions Distribution')
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Exploration metrics
        self._plot_exploration_metrics(ax4)
        
        plt.tight_layout()
        self._save_plot(fig, filename, save_plots)
        plt.show()

    def _plot_2d_summary(self, filename=None, save_plots=False, plot_type='both'):
        """2D summary plots (potential surfaces, trajectory, exploration)"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Original potential
        (X, Y), _ = self.compute_free_energy_surface()
        Z_orig = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_orig[i,j] = self.abc.potential.potential(np.array([X[i,j], Y[i,j]]))
        
        levels = self._log_levels(Z_orig, 50)
        contour1 = ax1.contour(X, Y, Z_orig, levels=levels, colors='black', alpha=0.6)
        ax1.clabel(contour1, inline=True, fontsize=8)
        
        # Mark known minima and saddles
        if hasattr(self.abc.potential, 'known_minima'):
            minima = self.abc.potential.known_minima()
            if minima is not None:
                ax1.scatter([m[0] for m in minima], [m[1] for m in minima],
                          c='g', marker='o', s=50, label='Minima')
        
        if hasattr(self.abc.potential, 'known_saddles'):
            saddles = self.abc.potential.known_saddles()
            if saddles is not None:
                ax1.scatter([s[0] for s in saddles], [s[1] for s in saddles],
                          c='purple', marker='o', s=50, label='Saddles')
        
        ax1.set_title('Original Potential')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Biased potential with bias centers
        (X_bias, Y_bias), Z_bias = self.compute_free_energy_surface()
        contour2 = ax2.contour(X_bias, Y_bias, Z_bias, levels=self._log_levels(Z_bias, 50), 
                              colors='blue', alpha=0.6)
        ax2.clabel(contour2, inline=True, fontsize=8)
        
        # Mark bias centers
        bias_centers = [b.center for b in self.abc.bias_list]
        ax2.scatter([c[0] for c in bias_centers], [c[1] for c in bias_centers],
                   c='red', s=5, marker='x', label='Bias Centers')
        
        # Mark found minima and saddles
        ax2.scatter([m[0] for m in self.abc.minima], [m[1] for m in self.abc.minima],
                   c='green', marker='o', s=50, label='Found Minima')
        ax2.scatter([s[0] for s in self.abc.saddles], [s[1] for s in self.abc.saddles],
                   c='purple', marker='o', s=50, label='Found Saddles')
        
        ax2.set_title('Biased Potential Surface')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trajectory with markers
        trajectory = self.get_trajectory()
        points = np.array(trajectory).reshape(-1,1,2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, cmap='viridis', 
                          array=np.linspace(0, len(trajectory), len(segments)),
                          linewidth=2, alpha=0.8)
        ax3.add_collection(lc)
        
        # Mark biases and perturbations
        bias_steps = self.get_bias_steps()
        if bias_steps.size > 0 and plot_type in ['biases', 'both']:
            bias_pos = np.array(trajectory)[bias_steps]
            ax3.scatter(bias_pos[:,0], bias_pos[:,1], c='red', s=4, 
                       marker='x', label='Biases', zorder=5)
        
        pert_steps = self.get_perturbation_steps()
        if pert_steps.size > 0 and plot_type in ['perturbations', 'both']:
            pert_pos = np.array(trajectory)[pert_steps]
            ax3.scatter(pert_pos[:,0], pert_pos[:,1], c='blue', s=4, 
                       marker='o', label='Perturbations', zorder=5)
        
        ax3.set_title('ABC Trajectory')
        ax3.legend()
        plt.colorbar(lc, ax=ax3, label='Time Step')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Exploration metrics
        self._plot_exploration_metrics(ax4)
        
        plt.tight_layout()
        self._save_plot(fig, filename, save_plots)
        plt.show()

    def plot_diagnostics(self, filename=None, save_plots=False, plot_type='Neither'):
        """Generate diagnostic plots (force, energy, barriers, exploration metrics)"""
        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(2, 2)  # 2 rows, 2 columns
        
        # Create subplots in the grid
        ax0 = fig.add_subplot(gs[0, 0])  # Top left - Force
        ax1 = fig.add_subplot(gs[0, 1])  # Top right - Energy
        ax2 = fig.add_subplot(gs[1, 0])  # Bottom left - Exploration
        ax3 = fig.add_subplot(gs[1, 1])  # Bottom right - Barriers
        
        # Plot 1: Force magnitude (top left)
        biased_forces = self.get_forces(biased=True)
        biased_mags = np.linalg.norm(biased_forces, axis=1) if biased_forces.ndim > 1 else np.abs(biased_forces)
        ax0.plot(biased_mags, 'c-', alpha=0.7, label='Biased Force')

        forces = self.get_forces(biased=False)
        if forces is not None and forces[0] is not None: 
            force_mags = np.linalg.norm(forces, axis=1) if forces.ndim > 1 else np.abs(forces)
            ax0.plot(force_mags, 'b-', label='Unbiased Force')
        
        ax0.set_title('Force Magnitude Over Time')
        ax0.set_ylabel('|Force|')
        ax0.legend()
        ax0.grid(True, alpha=0.3)
        
        # Plot 2: Energy diagnostics (top right)
        energies = self.get_energies(biased=False)
        ax1.plot(energies, color='orange', label='Unbiased PES')
        biased_energies = self.get_energies(biased=True)
        ax1.plot(biased_energies, color='blue', alpha=0.5, label='Biased PES')
        
        ax1.set_title('Energy Profile')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Energy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Mark biases and perturbations
        bias_steps = self.get_bias_steps()
        if bias_steps.size > 0 and plot_type in ['biases', 'both']:
            ax0.scatter(bias_steps, biased_mags[bias_steps], c='red', s=8, label='Biases')
            ax1.scatter(bias_steps, energies[bias_steps], c='red', s=8, label='Biases')
        
        pert_steps = self.get_perturbation_steps()
        if pert_steps.size > 0 and plot_type in ['perturbations', 'both']:
            ax0.scatter(pert_steps, biased_mags[pert_steps], c='blue', s=8, label='Perturbations')
            ax1.scatter(pert_steps, energies[pert_steps], c='blue', s=8, label='Perturbations')

        
        # Plot 3: Exploration metrics (bottom left)
        self._plot_exploration_metrics(ax2)
        
        # Plot 4: Barrier heights between minima (bottom right)
        ax3.set_title('Barrier Heights Between Minima')
        ax3.set_ylabel('Energy')
        ax3.grid(True, alpha=0.3)

        if (hasattr(self.abc, 'minima') and len(self.abc.minima) > 1 and 
            len(energies) > 0 and hasattr(self, 'get_trajectory')):
            
            try:
                trajectory = self.get_trajectory()
                # print(f"Trajectory length: {len(trajectory)}")
                
                minima_indices = [
                    np.argmin(np.linalg.norm(trajectory - min_pos, axis=1))
                    for min_pos in self.abc.minima
                ]
                # print(f"Minima indices: {minima_indices}")

                saddle_indices = [
                    np.argmin(np.linalg.norm(trajectory - saddle_pos, axis=1))
                    for saddle_pos in self.abc.saddles
                ]
                
                barriers = []
                labels = []
                for i in range(len(minima_indices)-1):
                    start, end = minima_indices[i], minima_indices[i+1]
                    max_energy = np.max(biased_energies[start:end])
                    barrier = max_energy - energies[start]
                    barriers.append(barrier)
                    labels.append(f'{i}→{i+1}')
                    other_barrier = max_energy - energies[end]
                    barriers.append(other_barrier)
                    labels.append(f'{i+1}→{i}')
                
                
                barriers = np.array(barriers).flatten()
                # print(f"Barriers: {barriers}")
                # print(f"Labels: {labels}")

                
                if len(barriers) > 0 and len(barriers) == len(labels):
                    ax3.bar(labels, barriers, width=0.6)
                    # Set reasonable y-limits if we have valid barriers
                    ymin = min(0, min(barriers)*1.1)
                    ymax = max(barriers)*1.1
                    ax3.set_ylim(ymin, ymax)
                else:
                    # Set default view for text
                    ax3.set_xlim(-0.5, 0.5)
                    ax3.set_ylim(-0.5, 0.5)
                    ax3.text(0, 0, 'No valid barriers found', 
                            ha='center', va='center', fontsize=12)
                    

                # Mark minima steps on energy plot
                for i, step in enumerate(minima_indices):
                    if step < len(energies):
                        ax1.scatter(step, energies[step], c='green', s=50, marker='*', 
                                label='Minima' if i == 0 else None, zorder=5)
                        

                for i, step in enumerate(saddle_indices):
                    if step < len(energies):
                        ax1.scatter(step, energies[step], c='purple', s=50, marker='*', 
                                label='Saddles' if i == 0 else None, zorder=5)
                        
            except Exception as e:
                print(f"Error calculating barriers: {str(e)}")
                # Set default view for text
                ax3.set_xlim(-0.5, 0.5)
                ax3.set_ylim(-0.5, 0.5)
                ax3.text(0, 0, f'Error plotting barriers:\n{str(e)}', 
                        ha='center', va='center', fontsize=12)
        else:
            # Set default view for text
            ax3.set_xlim(-0.5, 0.5)
            ax3.set_ylim(-0.5, 0.5)
            ax3.text(0, 0, 'Not enough minima/energy data', 
                    ha='center', va='center', fontsize=12)
            
        plt.tight_layout()
        self._save_plot(fig, filename, save_plots)
        plt.show()

    def _plot_exploration_metrics(self, ax):
        """Plot exploration metrics (distance from start, rolling variance)"""
        trajectory = self.get_trajectory()
        distances = cdist(trajectory, [trajectory[0]]).flatten()
        ax.plot(distances, 'b-', label='Distance from Start')
        
        window_size = min(3, len(trajectory)//3)
        variances = []
        for i in range(len(trajectory)-window_size):
            variances.append(np.sum(np.var(trajectory[i:i+window_size], axis=0)))
        
        ax.plot(np.arange(window_size, len(trajectory)), variances, 
              'g-', label=f'Rolling Var (w={window_size})')
        
        ax.set_title('Exploration Metrics')
        ax.set_ylabel('Metric Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Legacy functions
    @staticmethod
    def plot_results(abc_sim, filename=None, save_plots=False):
        """Legacy function - creates summary plots from ABC simulation"""
        analyzer = ABCAnalysis(abc_sim)
        analyzer.plot_summary(filename, save_plots)

    @staticmethod
    def analyze_basin_visits(abc_sim, basin_radius=0.3, verbose=True):
        """Legacy function - analyzes minima visits"""
        analyzer = ABCAnalysis(abc_sim)
        return analyzer.analyze_minima_saddles(proximity_radius=basin_radius)
    
  
    def create_basin_filling_gif(self, filename="basin_filling.gif", fps=15):
        """
        SIMPLE basin filling animation - shows potential with accumulating biases
        """
        fig, ax = plt.subplots(figsize=(8,5))
        from matplotlib.animation import FuncAnimation, PillowWriter
        
        # Get data
        x = np.linspace(*self.abc.potential.plot_range(), 500)
        orig_potential = self.abc.potential.potential(x)
        n_frames = len(self.abc.bias_list)
        
        # Initial plot
        ax.plot(x, orig_potential, 'k-', label='Original Potential')
        line, = ax.plot(x, orig_potential, 'b-', label='Biased Potential')
        ax.set_title('Potential Energy Landscape')
        ax.set_xlabel('x')
        ax.set_ylabel('Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Simple update function
        def update(frame):
            # Calculate current potential
            F = orig_potential.copy()
            for bias in self.abc.bias_list[:frame+1]:
                F += bias.potential(x[:,None])
            
            # Update line
            line.set_ydata(F)            
            return [line]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
        anim.save(filename, writer='pillow', fps=fps)
        plt.close()
        
        print(f"Saved animation to {filename}")