import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.distance import cdist
import h5py

class ABCAnalysis:
    """
    Streamlined analysis toolkit that maintains original plotting functionality
    while properly separating concerns. All storage access happens through the
    storage interface only.
    """

    def __init__(self, source):
        """
        Initialize with either:
        - ABC simulation object (TraditionalABC or SmartABC)
        - ABCStorage instance
        - Path to storage directory
        
        Args:
            source: Simulation object, storage instance, or path string
        """
        if isinstance(source, str):
            from storage import ABCStorage  # Avoid circular import
            self.storage = ABCStorage(source)
            self.from_storage = True
        elif hasattr(source, 'dump_current'):  # ABCStorage instance
            self.storage = source
            self.from_storage = True
        else:  # ABC simulation instance
            from storage import ABCStorage
            self.storage = ABCStorage()  # Default storage
            self.abc = source
            self.from_storage = False
            
        self._init_dimension()
        self._load_persistent_data()
        
    def _init_dimension(self):
        """Initialize system dimension from available data"""
        if not self.from_storage:
            self.dimension = self.abc.dimension
            return
            
        try:
            # Get dimension from first iteration
            iter_data = self.storage.load_iteration(0)
            positions = iter_data['history']['positions']
            self.dimension = positions.shape[1] if len(positions.shape) > 1 else 1
        except:
            self.dimension = None  # Will be set when data becomes available
    
    def _load_persistent_data(self):
        """Load biases and landmarks from storage"""
        if not self.from_storage:
            return
            
        # Initialize empty attributes that would exist in live simulation
        self.abc = type('DummyABC', (), {})()  # Empty object
        self.abc.minima = []
        self.abc.saddles = []
        self.abc.bias_list = []
        
        # Try to load from most recent iteration
        try:
            last_iter = self.storage.get_most_recent_iter()
            if last_iter is not None:
                iter_data = self.storage.load_iteration(last_iter)
                self.abc.minima = iter_data.get('minima', [])
                self.abc.saddles = iter_data.get('saddles', [])
                self.abc.bias_list = iter_data.get('biases', [])
        except:
            pass

    # Data access methods (storage interface only)
    def get_trajectory(self):
        """Get complete trajectory"""
        if not self.from_storage:
            return np.array(self.abc.trajectory)
        return np.concatenate([
            data['history']['positions'] 
            for data in self._load_all_iterations()
        ])

    def get_energies(self, biased=False):
        """Get energy values"""
        key = 'biased_energies' if biased else 'unbiased_energies'
        if not self.from_storage:
            return np.array(getattr(self.abc, key, []))
        return np.concatenate([
            data['history'][key] 
            for data in self._load_all_iterations()
        ])

    def get_forces(self, biased=False):
        """Get force values"""
        key = 'biased_forces' if biased else 'unbiased_forces'
        if not self.from_storage:
            return np.array(getattr(self.abc, key, []))
        return np.concatenate([
            data['history'][key] 
            for data in self._load_all_iterations()
        ])

    def _load_all_iterations(self):
        """Generator yielding data from all iterations"""
        if not self.from_storage:
            return []
            
        for filename in self.storage.get_all_files():
            with h5py.File(filename, 'r') as f:
                for iter_key in sorted(f.keys(), key=lambda x: int(x.split('_')[1])):
                    yield self.storage.load_iteration(int(iter_key.split('_')[1]))

    def get_bias_steps(self):
        """Identify steps where biases were placed"""
        if not self.from_storage:
            if not hasattr(self.abc, 'iter_periods'):
                return np.array([], dtype=int)
            periods = np.array(self.abc.iter_periods)
        else:
            periods = []
            for data in self._load_all_iterations():
                periods.append(len(data['history']['positions']))
                # TODO: use persistent stored iter_period var in landmarks 
            periods = np.array(periods)
        
        def get_end_indices(lengths):
            """
            Given a list of lengths of sublists, return a list of indices corresponding to the
            last element of each sublist in the overall concatenated list.

            Example:
                lengths = [3, 5, 2]
                => sublists: [0,1,2], [3,4,5,6,7], [8,9]
                => returns: [2, 7, 9]
            """
            end_indices = []
            current_index = -1
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
        else: 
            return bias_steps[:-1]+1 # 1 fewer perturb step than bias steps (no perturbation at the end)

    # Original plotting methods (exactly as before)
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

    def plot_summary(self, filename=None, save_plots=False, plot_type='both'):
        """Generate standard summary plots appropriate for system dimension"""
        if self.dimension == 1:
            self._plot_1d_summary(filename, save_plots, plot_type)
        elif self.dimension == 2:
            self._plot_2d_summary(filename, save_plots, plot_type)
        else:
            print(f"Standard visualization not supported for {self.dimension}D systems")
            self.plot_diagnostics(plot_type=plot_type)

    def _plot_1d_summary(self, filename=None, save_plots=False, plot_type='both'):
        """1D summary plots (potential, trajectory, histogram, energy profile)"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Potential landscape
        if not self.from_storage and hasattr(self.abc, 'potential'):
            x, F_orig = self.abc.compute_free_energy_surface(resolution=1000)
            ax1.plot(x, self.abc.potential.potential(x), 'k-', label='Original Potential')
            ax1.plot(x, F_orig, 'b-', alpha=0.7, label='Biased Potential')
        
        # Mark minima and saddles
        if hasattr(self.abc, 'minima'):
            for i, min_pos in enumerate(self.abc.minima):
                ax1.axvline(min_pos[0], color='g', linestyle='--', alpha=0.5, label='Minima' if i == 0 else None)
        if hasattr(self.abc, 'saddles'):
            for i, sad_pos in enumerate(self.abc.saddles):
                ax1.axvline(sad_pos[0], color='orange', linestyle=':', alpha=0.5, label='Saddles' if i == 0 else None)
        # Mark biases in red
        if hasattr(self.abc, 'bias_list') and self.abc.bias_list:
            for i, bias in enumerate(self.abc.bias_list):
                ax1.axvline(bias.center[0], color='red', linestyle='-', alpha=0.5, label='Biases' if i == 0 else None)
        
        ax1.set_title('Potential Energy Landscape')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Energy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Trajectory with markers
        trajectory = self.get_trajectory()
        times = np.arange(len(trajectory))
        ax2.plot(times, trajectory, 'g-', alpha=0.7, linewidth=1)
        
        # Mark biases and perturbations
        if plot_type in ['biases', 'both']:
            bias_steps = self.get_bias_steps()
            if bias_steps.size > 0:
                ax2.scatter(bias_steps, trajectory[bias_steps], c='red', s=20, marker='x', label='Biases')
        if plot_type in ['perturbations', 'both']:
            pert_steps = self.get_perturbation_steps()
            if pert_steps.size > 0:
                ax2.scatter(pert_steps, trajectory[pert_steps], c='blue', s=20, marker='o', label='Perturbations')
        
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
        
        # Plot 4: Exploration metrics
        self._plot_exploration_metrics(ax4)
        
        plt.tight_layout()
        self._save_plot(fig, filename, save_plots)
        plt.show()

    def _plot_2d_summary(self, filename=None, save_plots=False, plot_type='both'):
        """2D summary plots (potential surfaces, trajectory, exploration)"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Original potential
        if not self.from_storage and hasattr(self.abc, 'potential'):
            (X, Y), _ = self.abc.compute_free_energy_surface()
            Z_orig = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z_orig[i,j] = self.abc.potential.potential(np.array([X[i,j], Y[i,j]]))
            
            levels = self._log_levels(Z_orig, 50)
            contour1 = ax1.contour(X, Y, Z_orig, levels=levels, colors='black', alpha=0.6)
            ax1.clabel(contour1, inline=True, fontsize=8)
        
        # Mark minima and saddles
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
        if not self.from_storage and hasattr(self.abc, 'compute_free_energy_surface'):
            (X_bias, Y_bias), Z_bias = self.abc.compute_free_energy_surface()
            contour2 = ax2.contour(X_bias, Y_bias, Z_bias, levels=self._log_levels(Z_orig, 50), 
                                  colors='blue', alpha=0.6)
            ax2.clabel(contour2, inline=True, fontsize=8)
        
        if hasattr(self.abc, 'bias_list') and self.abc.bias_list:
            bias_centers = [b.center for b in self.abc.bias_list]
            ax2.scatter([c[0] for c in bias_centers], [c[1] for c in bias_centers],
                       c='red', s=50, marker='x', label='Bias Centers')
            ax2.legend()
        
        ax2.set_title('Biased Potential Surface')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trajectory with markers
        trajectory = self.get_trajectory()
        if len(trajectory) > 1:
            points = np.array(trajectory).reshape(-1,1,2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            lc = LineCollection(segments, cmap='viridis', 
                              array=np.linspace(0, len(trajectory), len(segments)),
                              linewidth=2, alpha=0.8)
            ax3.add_collection(lc)
            
            # Mark biases and perturbations
            if plot_type in ['biases', 'both']:
                bias_steps = self.get_bias_steps()
                if bias_steps.size > 0:
                    bias_pos = np.array(trajectory)[bias_steps]
                    ax3.scatter(bias_pos[:,0], bias_pos[:,1], c='red', s=50, 
                               marker='x', label='Biases', zorder=5)
            if plot_type in ['perturbations', 'both']:
                pert_steps = self.get_perturbation_steps()
                if pert_steps.size > 0:
                    pert_pos = np.array(trajectory)[pert_steps]
                    ax3.scatter(pert_pos[:,0], pert_pos[:,1], c='blue', s=50, 
                               marker='o', label='Perturbations', zorder=5)
            
            ax3.set_title('ABC Trajectory')
            ax3.legend()
            plt.colorbar(lc, ax=ax3, label='Time Step')
        else:
            ax3.text(0.5, 0.5, 'No trajectory data', ha='center', va='center')
        
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Exploration metrics
        self._plot_exploration_metrics(ax4)
        
        plt.tight_layout()
        self._save_plot(fig, filename, save_plots)
        plt.show()

    def plot_diagnostics(self, filename=None, save_plots=False, plot_type='both'):
        """Generate diagnostic plots (force, energy, barriers)"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))                
        
        # Plot 1: Force magnitude
        biased_forces = self.get_forces(biased=True)
        if len(biased_forces) > 0:
            biased_mags = np.linalg.norm(biased_forces, axis=1) if biased_forces.ndim > 1 else np.abs(biased_forces)
            axes[0].plot(biased_mags, 'c-', alpha=0.7, label='Biased Force')

        forces = self.get_forces(biased=False)
        if len(forces) > 0 and forces[0] is not None:
            force_mags = np.linalg.norm(forces, axis=1) if forces.ndim > 1 else np.abs(forces)
            axes[0].plot(force_mags, 'b-', label='Unbiased Force')
        
        axes[0].set_title('Force Magnitude Over Time')
        axes[0].set_ylabel('|Force|')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Energy diagnostics
        energies = self.get_energies(biased=False)
        if len(energies) > 0:
            axes[1].plot(energies, color='orange', label='Unbiased PES')
            
            biased_energies = self.get_energies(biased=True)
            if len(biased_energies) > 0:
                axes[1].plot(biased_energies, color='blue', alpha=0.5, label='Biased PES')
        
        axes[1].set_title('Energy Profile')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Energy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Mark biases and perturbations
        if plot_type in ['biases', 'both']:
            bias_steps = self.get_bias_steps()
            if bias_steps.size > 0 and len(biased_mags) > 0:
                axes[0].scatter(bias_steps, biased_mags[bias_steps], c='red', s=30, label='Biases')
            if bias_steps.size > 0 and len(energies) > 0:
                axes[1].scatter(bias_steps, energies[bias_steps], c='red', s=30, label='Biases')
        
        if plot_type in ['perturbations', 'both']:
            pert_steps = self.get_perturbation_steps()
            if pert_steps.size > 0 and len(forces) > 0:
                axes[0].scatter(pert_steps, biased_mags[pert_steps], c='blue', s=30, label='Perturbations')
            if pert_steps.size > 0 and len(energies) > 0:
                axes[1].scatter(pert_steps, energies[pert_steps], c='blue', s=30, label='Perturbations')
        
        # Plot 3: Barrier heights between minima
        if hasattr(self.abc, 'minima') and len(self.abc.minima) > 1 and len(energies) > 0:
            minima_indices = [
                np.argmin(np.linalg.norm(self.get_trajectory() - min_pos, axis=1))
                for min_pos in self.abc.minima
            ]
            
            barriers = []
            labels = []
            for i in range(len(minima_indices)-1):
                start, end = minima_indices[i], minima_indices[i+1]
                max_energy = np.max(energies[start:end])
                barrier = max_energy - min(energies[start], energies[end])
                barriers.append(barrier)
                labels.append(f'{i}→{i+1}')
            
            if barriers:
                axes[2].bar(labels, barriers)
                axes[2].set_title('Barrier Heights Between Minima')
                axes[2].set_ylabel('Energy')
                axes[2].grid(True, alpha=0.3)
            else:
                axes[2].text(0.5, 0.5, 'No barrier data', ha='center', va='center')
        else:
            axes[2].text(0.5, 0.5, 'Not enough minima/energy data', ha='center', va='center')
        
        plt.tight_layout()
        self._save_plot(fig, filename, save_plots)
        plt.show()

    def _plot_exploration_metrics(self, ax):
        """Plot exploration metrics (distance from start, rolling variance)"""
        trajectory = self.get_trajectory()
        if len(trajectory) > 2:
            distances = cdist(trajectory, [trajectory[0]]).flatten()
            ax.plot(distances, 'b-', label='Distance from Start')
            
            window_size = min(3, len(trajectory)//3)
            variances = []
            for i in range(len(trajectory)-window_size):
                variances.append(np.sum(np.var(trajectory[i:i+window_size], axis=0)))
            
            ax.plot(np.arange(window_size, len(trajectory)), variances, 
                   'g-', label=f'Rolling Var (w={window_size})')
            
            ax.set_title('Exploration Metrics')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')

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