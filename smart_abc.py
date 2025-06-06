import numpy as np
from scipy.optimize import minimize
from storage import ABCStorage 
from bias import GaussianBias
from optimizer import ScipyOptimizer, ASEOptimizer

__author__ = "Simon Nirenberg"
__email__ = "simon_nirenberg@brown.edu"
    
class SmartABC:
    """
    Autonomous Basin Climbing (ABC) algorithm with smart perturbation and biasing strategies.
    This class has all the functionaly of TraditionalABC and much more. 
    """

    def __init__(
        self,
        potential,
        expected_barrier_height=1,
        # Optional parameters organized by category
        # Setup parameters
        run_mode="compromise",
        starting_position=None,
        dump_every = 50, # Iters between data dumps
        dump_folder = "abc_data_dumps", 

        # Curvature estimation
        curvature_method="full_hessian",
        
        # Perturbation strategy
        perturb_type="dynamic",
        perturb_dist="normal",
        scale_perturb_by_curvature=True,
        default_perturbation_size=0.05,
        large_perturbation_scale_factor=5,
        
        # Biasing strategy
        bias_type="smart",
        default_bias_height=None,
        default_bias_covariance=None,
        curvature_bias_height_scale=None,
        curvature_bias_covariance_scale=None,
        
        # Descent and optimization
        optimizer = None, 
        descent_convergence_threshold=1e-5,
        max_descent_steps=20,
        max_descent_step_size=1.0,
        max_acceptable_force_mag = 500.
    ):
        """
        Initialize the SmartABC sampler.
        
        Args:
            potential: Potential energy surface to sample
            expected_barrier_height: Estimated average barrier height (for scaling)
            
            See README.md for full documentation of optional parameters.
        """
        self.potential = potential
        self.expected_barrier_height = expected_barrier_height
        
        # Set up configuration parameters
        self.run_mode = run_mode
        self.curvature_method = curvature_method

        self.dump_every = dump_every
        self.dump_folder = dump_folder 
        if dump_every != 0 and dump_folder is not None:
            self.storage = ABCStorage(dump_folder, dump_every) 
        else: 
            self.storage = None 
        
        self.perturb_type = perturb_type
        self.perturb_dist = perturb_dist
        self.scale_perturb_by_curvature = scale_perturb_by_curvature
        self.default_perturbation_size = default_perturbation_size
        self.large_perturbation_scale_factor = large_perturbation_scale_factor
        
        self.bias_type = bias_type
        self.default_bias_height = default_bias_height or expected_barrier_height
        self.default_bias_covariance = default_bias_covariance or 1.0
        self.curvature_bias_height_scale = curvature_bias_height_scale or 1.0
        self.curvature_bias_covariance_scale = curvature_bias_covariance_scale or 1.0
        
        if optimizer is None:
            optimizer = ScipyOptimizer(self)
        self.optimizer = optimizer
        self.descent_convergence_threshold = descent_convergence_threshold
        self.max_descent_steps = max_descent_steps
        self.max_descent_step_size = max_descent_step_size
        self.max_acceptable_force_mag = max_acceptable_force_mag
        self.potential.max_acceptable_force_mag = max_acceptable_force_mag
        
        # Automatically validate user input to minimize chance of conflicts later
        self.validate_config()

        # Initialize state variables
        self.reset(starting_position)

    def reset(self, starting_position=None):
        """Reset the sampler to initial state."""
        if starting_position is None:
            self.position = self.potential.default_starting_position()
        else:
            self.position = np.array(starting_position, dtype=float)
            
        self.bias_list = []
        self.trajectory = []
        self.biased_energies = []
        self.unbiased_energies = []
        self.biased_forces = []
        self.unbiased_forces = []

        self.minima = []
        self.saddles = []

        self._dimension = len(self.position)

        self.prev_mode = None
        self.most_recent_hessian = None
        self.most_recent_inv_hessian = None # Only used with curvature_method = "bfgs"

        self.current_iteration = 0
        self.iter_periods = [] 

    @property
    def dimension(self):
        return self._dimension
    
    def validate_config(self):
        """Validate configuration parameters."""
        # TODO: Implement comprehensive validation
        print("Config Valid")

    @classmethod
    def load_from_disk(cls, folder_path, *args, **kwargs):
        """Loads ABC instance from disk. Passes ABC creation args and kwargs to __init__."""
        storage = ABCStorage(folder_path)
        iteration = storage.get_most_recent_iter()
        if iteration is None:
            raise RuntimeError("No iterations found in storage.")
        data = storage.load_iteration(iteration)
        pos = data['history']['positions'][-1]
        abc = cls(starting_position=pos, *args, **kwargs)
        abc.current_iteration = iteration + 1
        abc.bias_list = data['biases']
        return abc 
    
    def clear_history_lists(self):
        """
        Clears possibly lengthy lists stored in RAM
        I strongly recommend only ever using this after first calling dump_data()

        Does NOT clear / reset non-list state variables (e.g. most_recent_hessian)
        """
        self.trajectory.clear()
        self.unbiased_energies.clear()
        self.biased_energies.clear()
        self.unbiased_forces.clear()
        self.biased_forces.clear()
        self.iter_periods.clear

    def update_records(self):
        """Update iteration number and dump history if it is time to do so"""
        period = len(self.trajectory) - np.sum(self.iter_periods)
        self.iter_periods.append(period)

        if self.storage and self.current_iteration % self.dump_every == 0:
            self.storage.dump_current(self)
            self.clear_history_lists() 
        self.current_iteration += 1

    # Core functionality (adapted from TraditionalABC with enhancements)
    
    def compute_biased_potential(self, position, unbiased: float = None) -> float:
        """Compute total potential including biases."""
        if unbiased is None: 
            V = self.potential.potential(position)
        else: 
            V = unbiased
        for bias in self.bias_list:
            V += bias.potential(position)
        
        return V

    def compute_biased_force(self, position, unbiased: np.ndarray = None, eps=1e-5) -> np.ndarray:
        """Compute total force using analytical gradients or finite differences."""
        if unbiased is not None: 
            total_grad = - unbiased.copy()
            for bias in self.bias_list:
                total_grad += bias.gradient(position)
            force = -total_grad     
            
        else: 
            try:
                unbiased = - self.potential.gradient(position)
            except NotImplementedError:          
                # Fall back to finite difference
                force = np.zeros_like(position)
                for i in range(len(position)):
                    pos_plus = position.copy()
                    pos_minus = position.copy()
                    pos_plus[i] += eps
                    pos_minus[i] -= eps
                    force[i] = -(self.compute_biased_potential(pos_plus).item() - self.compute_biased_potential(pos_minus).item()) / (2 * eps)

        if (norm := np.linalg.norm(force)) > self.max_acceptable_force_mag: 
            print(f"Warning: Force value of {force} detected as likely unphysically large in magnitude; shrunk to magnitude {self.max_acceptable_force_mag}")
            force = self.max_acceptable_force_mag * force / norm
            
        return force

    def deposit_bias(self, center: np.ndarray = None, covariance: float|np.ndarray = None, height: float = None):
        """Deposit a new Gaussian bias potential.

        Args:
            center: Center of the bias (defaults to current position).
            covariance: Covariance (float for isotropic, or np.ndarray for anisotropic).
            height: Height of the bias.
        """
        pos = center if center is not None else self.position.copy()
        
        # Smart bias scaling based on curvature if available
        if self.bias_type == "smart" and self.most_recent_hessian is not None:
            if height is None:
                height = self.curvature_bias_height_scale * self.expected_barrier_height
            if covariance is None:
                # Scale covariance by inverse curvature
                cov = self.curvature_bias_covariance_scale * np.linalg.inv(
                    self.make_positive_definite(self.most_recent_hessian)
                )
        else:
            # Fall back to defaults
            if height is None:
                height = self.default_bias_height
            if covariance is None:
                cov = np.square(self.default_bias_sigma) if hasattr(self, 'default_bias_sigma') else self.default_bias_covariance
                # backwards compatibility 

        bias = GaussianBias(
            center=pos,
            covariance=cov,
            height=height
        )
        self.bias_list.append(bias)
        print(f"Deposited bias at {pos} with (co)variance {cov} and height {height}.")

    def descend(self, max_steps=None, convergence_threshold=None):
        """
        Efficient descent with built-in state recording that avoids callbacks.
        All force and energy calculations are done exactly once per step.
        """
        max_steps = max_steps or self.max_descent_steps
        convergence_threshold = convergence_threshold or self.descent_convergence_threshold

        result, traj_data = self.optimizer.descend(self.position, max_steps=self.max_descent_steps, convergence_threshold=self.descent_convergence_threshold)
        final_pos = result['x']
        converged = result['converged']
        hess_inv = result['hess_inv'] if 'hess_inv' in result else None
        traj, unbiased_e, biased_e, unbiased_f, biased_f = traj_data 

        self.trajectory.extend(traj)
        self.unbiased_energies.extend(unbiased_e)
        self.biased_energies.extend(biased_e)
        self.unbiased_forces.extend(unbiased_f)
        self.biased_forces.extend(biased_f)

        # # Ensure final state is recorded (in case optimizer didn't call our wrapper)
        # final_pos = result.x.copy()
        # if len(self.trajectory) == 0 or not np.allclose(self.trajectory[-1], final_pos):
        #     self._record_state(final_pos)

        # Process curvature and check minimum (same as before)
        self._process_curvature_info(final_pos, hess_inv)
        self._check_minimum(converged, final_pos, convergence_threshold)

        self.position = final_pos.copy()
        return converged

    # Curvature util

    def _process_curvature_info(self, final_pos, hess_inv=None):
        """Process curvature information from optimization result."""
        if self.curvature_method == "full_hessian":
            self.most_recent_hessian = self.compute_hessian_finite_difference(
                final_pos, 
                f0_already_computed=True
            )

        if hess_inv is not None:
            self.most_recent_inv_hessian = hess_inv

    def _check_minimum(self, converged, final_pos, threshold):
        """Check if the final position is a minimum."""
        if converged:
            # print(len(self.biased_energies))
            # print(len(self.biased_forces))
            if self.unbiased_forces[-1] is None:
                unbiased_force = self.biased_forces[-1].copy()
                for bias in self.bias_list:
                    unbiased_force -= (-bias.gradient(final_pos))
            else:
                unbiased_force = self.unbiased_forces[-1]
                
            if np.isclose(np.linalg.norm(unbiased_force), 0, threshold):
                if self.most_recent_hessian is not None: 
                    unbiased_pes_hessian = self.most_recent_hessian
                elif self.most_recent_inv_hessian is not None: 
                    try:
                        unbiased_pes_hessian = np.linalg.inv(self.most_recent_inv_hessian)
                    except np.linalg.LinAlgError:
                        print("Warning: Could not invert inverse Hessian - skipping curvature analysis.")
                        unbiased_pes_hessian = None

                if unbiased_pes_hessian is not None: 
                    for bias in self.bias_list:
                        unbiased_pes_hessian -= bias.hessian(final_pos)
                    crit_type = self.evaluate_critical_point(final_pos, unbiased_pes_hessian)
                    if crit_type == "minimum": 
                        self.minima.append(final_pos.copy())
                    elif crit_type == "saddle":
                        self.saddles.append(final_pos.copy())
                else: 
                    self.minima.append(final_pos.copy())

    def compute_hessian_finite_difference(self, position, eps=1e-3, f0_already_computed=False):
        n = len(position)
        hessian = np.zeros((n, n))
        if f0_already_computed:
            f0 = self.biased_forces[-1]
        else: 
            f0 = self.compute_biased_force(position)  # Cache f(position)
        
        
        # Cache forces at position + eps * e_i for all i
        f_i_cache = np.zeros((n, n))
        for i in range(n):
            delta_i = np.zeros(n)
            delta_i[i] = eps
            f_i_cache[i] = self.compute_biased_force(position + delta_i)
        
        # Compute upper triangle and diagonal
        for i in range(n):
            for j in range(i, n):
                delta_i = np.zeros(n)
                delta_j = np.zeros(n)
                delta_i[i] = eps
                delta_j[j] = eps

                f_ij = self.compute_biased_force(position + delta_i + delta_j)
                
                # Use cached f_i and f_j
                val = (f_ij[i] - f_i_cache[i][i] - f_i_cache[j][j] + f0[i]) / (eps ** 2)
                hessian[i, j] = val
                if i != j:
                    hessian[j, i] = val  # Exploit symmetry

        return -hessian  # negative because working with -∇V

    
    def evaluate_critical_point(self, hessian):
        """Classify critical point based on Hessian eigenvalues."""
        eigvals = np.linalg.eigvalsh(hessian)
        if np.all(eigvals > 0):
            return "minimum"
        elif np.all(eigvals < 0):
            return "maximum"
        else:
            return "saddle"
        
    def conjugate_directions(self, hessian):
        """
        Fully diagonalize the Hessian to find all conjugate directions and values.
        """
        eigvals, eigvecs = np.linalg.eigh(hessian)
        return zip(eigvecs.T, eigvals)  # transpose to convert to eigenvector-value pairs 
    
    def compute_exact_extreme_hessian_mode(self, hessian, desired_mode = "softest"):
        """
        Fully diagonalize the Hessian to find the exact softest mode.
        More efficient than estimator for low-dim potentials, worse for high-dim
        """
        conjugate_dir = list(self.conjugate_directions(hessian))
        sorted_dirs = sorted(conjugate_dir, key=lambda x: x[1])

        if desired_mode == "softest":
            return sorted_dirs[0]
        else:
            # returns hardest mode and curvature
            return sorted_dirs[-1]

    
    def estimate_softest_hessian_mode(self, position, init_direction=None, eps=1e-3, softmode_max_iters=25, tol=1e-4):
        """
        Estimate the softest eigenmode (lowest-curvature direction) without full Hessian,
        using the Rayleigh quotient.
        
        Returns:
            direction: Unit vector along softest mode
            curvature: Approximate eigenvalue
        """
        n = len(position)
        if init_direction is None:
            # Start with the previous minimum's softest mode if applicable; otherwise, start with a random unit vector
            direction = self.prev_mode if self.prev_mode is not None else np.random.randn(n)
        else:
            direction = np.array(init_direction)
        direction /= np.linalg.norm(direction)

        for _ in range(softmode_max_iters):
            # Finite-difference force approximation (central difference)
            f1 = self.compute_biased_force(position + eps * direction)
            f2 = self.compute_biased_force(position - eps * direction)

            # Effective Hessian-vector product: -H @ d ≈ (f1 - f2)/(2ε)
            h_d = (f1 - f2) / (2 * eps)

            # Rayleigh quotient: curvature ≈ dᵀ H d = -dᵀ h_d
            curvature = -np.dot(direction, h_d)

            # Gradient of curvature wrt direction is h_d - (dᵀ h_d) d
            # (i.e., remove component along d to keep normalization)
            grad = h_d - np.dot(h_d, direction) * direction

            # Gradient descent step to minimize curvature
            new_direction = direction - 0.1 * grad
            new_direction /= np.linalg.norm(new_direction)

            if np.linalg.norm(new_direction - direction) < tol:
                break

            direction = new_direction

        return direction, curvature
    
    
    def estimate_extreme_hessian_modes(self, position, n_min, n_max): 
        """
        Uses the Lanczos method to estimate the largest and smallest hessian modes.
        Cheaper than finite-difference but more expensive than Rayleigh. 
        """
        return NotImplementedError
        
    def get_softest_hessian_mode(self, position):
        """
        Gets the softest hessian mode and eigenvalue with maximal efficiency,
        accounting for the chosen curvature_method 
        """
        if self.most_recent_hessian is not None: 
            # Handles full_hessian since most_recent_hessian is set to 
            # appropriate hessian immediately when convergence is reached 
            hessian = self.most_recent_hessian
            return self.compute_exact_extreme_hessian_mode(hessian, desired_mode="softest")
        elif self.most_recent_inv_hessian is not None:
            # Handles bfgs method since softest mode of H is hardest of H_inv
            inv_hessian = self.most_recent_inv_hessian
            return self.compute_exact_extreme_hessian_mode(inv_hessian, desired_mode="hardest")
            
        elif self.curvature_method.lower() == "lanczos":
            mode, curvature = self.estimate_extreme_hessian_modes(position, 1, 0) 
            # TODO: may want to store lanczos to prevent calling again in bias 
            # deposition - some separate state variable only used with the lanczos 
            # mode.
        else:
            mode, curvature = self.estimate_softest_hessian_mode(position)
            # TODO: may want to store self.prev_mode here 
        return (mode, curvature)
       

    def make_positive_definite(self, H, eps=1e-6):
        """Modify Hessian to ensure positive definiteness."""
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals_mod = np.clip(eigvals, eps, None)
        return eigvecs @ np.diag(eigvals_mod) @ eigvecs.T


    # Perturbation and climbing

    def perturb(self, scale=None, mode="dynamic"):
        """
        Perturb current position using selected strategy.
        
        If mode == random or mode == dynamic and criterion fulfilled:
        random
        else:
        if fill_hessian available, compute its softest direction and perturb along it; otherwise, use rayleigh estimate for now 
        perturb_along_direction()
        Maybe make a new function to get this info regardless, to handle the hessian info type itself based on type so you dont have to specify 

        """
        if scale is None:
            scale = self.default_perturbation_size

        if mode == "random":
            self.perturb_random(scale)
        elif mode == "dynamic" and self.suspected_climbing_too_long():
            self.perturb_random(self.large_perturbation_scale_factor*scale)
        else:
            mode, curvature = self.get_softest_hessian_mode(self.position)
            if self.scale_perturb_by_curvature:
                mag = scale*mode 
            else: 
                mag = scale
            self.perturb_along_direction(curvature, mag)
        
        self._record_perturbation()
            

    def perturb_along_direction(self, direction, scale):
        """Perturb along specified direction."""
        direction = direction / np.linalg.norm(direction)
        self.position += scale * direction
        self._record_perturbation(f"Perturbed along direction to {self.position}")

    def perturb_random(self, scale):
        """Random Gaussian perturbation."""
        noise = np.random.normal(scale=scale, size=self.position.shape)
        self.position += noise
        self._record_perturbation(f"Randomly perturbed to {self.position}")

    def _record_perturbation(self, message=None):
        """Helper to record perturbation results."""
        if message is not None:
            print(message)
        # self._record_state(self.position)
        # self.trajectory.append(self.position.copy())
        # self.biased_forces.append(self.compute_biased_force(self.position))
        # self.biased_energies.append(self.compute_biased_potential(self.position))

    def suspected_climbing_too_long(self):
        # TODO: Implement
        return False 

    # Simulation control
    
    def run(self, max_iterations=100, verbose=True, **kwargs):
        """
        Run the ABC simulation.
        
        Args:
            max_iterations: Maximum number of iterations
            verbose: Whether to print progress
            kwargs: Override any default parameters
        """
        # Update parameters from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        for iteration in range(max_iterations):
            # Descent phase
            converged = self.descend()
            
            pos = self.position.copy() # position before perturbation

            # Compute Hessian if needed for smart biasing/perturbation
            if self.curvature_method == "full_hessian":
                self.most_recent_hessian = self.compute_hessian_finite_difference(self.position)
                               
            # Perturbation
            self.perturb(mode=self.perturb_type)

            # Deposit bias
            self.deposit_bias(pos)
            
            self.update_records()
            
            if verbose:
                print(f"Iteration {iteration+1}/{max_iterations}: "
                      f"Descent converged: {converged},"
                      f"Position {self.position}, "
                      f"Total biases: {len(self.bias_list)}")
        
        print(f"Simulation completed.\n"
              f"Identified minima: {self.minima},\n"
              f"On-the-fly-identified saddle-points (usually none: should find in post-processing with analyzer): {self.saddles}\n"
              f"Cached memory stats (use analyzer for dumped data):\n"
              f"\tTotal energy calls: {self.potential.energy_calls}\n"
              f"\tTotal force calls: {self.potential.force_calls}\n"
              f"\tTotal steps: {len(self.trajectory)}")

    # Analysis and visualization
    def get_trajectory(self):
        return np.array(self.trajectory)
        
    def get_biased_forces(self):
        return np.array(self.biased_forces)
        
    def get_biased_energies(self):
        return np.array(self.biased_energies)
        
    def get_bias_centers(self):
        return np.array([bias.center for bias in self.bias_list])
        
    def compute_free_energy_surface(self, resolution=100):
        """Compute free energy surface on grid for visualization."""
        if self.dimension == 1:
            x_range = self.potential.plot_range()
            x = np.linspace(x_range[0], x_range[1], resolution)
            F = np.array([self.compute_biased_potential(np.array([xi])) for xi in x])
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
                    F[i,j] = self.compute_biased_potential(pos)
            return (X, Y), F
        else:
            raise NotImplementedError("Visualization not implemented for dimensions > 2")
        

################################################################################
# Testing 

from potentials import Complex1D, DoubleWell1D, StandardMullerBrown2D
from analysis import ABCAnalysis

def run_1d_simulation():
    """Run 1D ABC simulation with complex potential."""
    np.random.seed(42)
    print("Starting 1D ABC Simulation")
    print("=" * 50)
    
    potential = Complex1D()
    abc = SmartABC(
        potential=potential,
        starting_position=[0.0],
        default_bias_height=1,
        default_bias_covariance=0.3,
        default_perturbation_size=0.05,
        optimizer=None, 
        run_mode="compromise",
        perturb_type="random",
        bias_type="constant",
        curvature_method="ignore",
        dump_folder=None, 
    )
    
    abc.run(max_iterations=10, verbose=True)
    
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total steps: {len(trajectory)}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: {abc.position}")
    
    # Create analysis and plots
    analyzer = ABCAnalysis(abc)
    # print(analyzer.analyze_minima_saddles(proximity_radius=0.1))
    analyzer.plot_summary(save_plots=True, filename="1d_smart_abc.png")
    analyzer.plot_diagnostics(save_plots=True, filename="1d_smart_abc_diagnostics.png")

def run_2d_simulation():
    """Run 2D ABC simulation with Muller-Brown potential."""
    np.random.seed(420)
    print("Starting 2D ABC Simulation")
    print("=" * 50)
    
    potential = StandardMullerBrown2D()
    abc = SmartABC(
        potential=potential,
        expected_barrier_height=30,  # Adjusted for Muller-Brown potential
        starting_position=[0.0, 0.0],
        optimizer=None,
        perturb_type="random",
        bias_type="constant",
        curvature_method="ignore",
        default_bias_height=15,
        default_bias_covariance=0.3,
        default_perturbation_size=0.2,
        max_descent_steps=100,
        dump_folder=None, 
    )
    
    abc.run(max_iterations=10, verbose=True)
    
    trajectory = abc.get_trajectory()
    # print(trajectory)
    print(f"\nSimulation Summary:")
    print(f"Total steps: {len(trajectory)}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: {abc.position}")
    
    # Create analysis and plots
    analyzer = ABCAnalysis(abc)
    # print(analyzer.analyze_minima_saddles(proximity_radius=0.1))
    analyzer.plot_summary(save_plots=True, filename="2d_smart_abc.png")
    analyzer.plot_diagnostics(save_plots=True, filename="2d_smart_abc_diagnostics.png")

def main():
    """Run both 1D and 2D simulations."""
    print("Running 1D Simulation")
    run_1d_simulation()
    
    print("\nRunning 2D Simulation")
    run_2d_simulation()

if __name__ == "__main__":
    main()