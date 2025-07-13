from copy import deepcopy
import numpy as np
from bias import GaussianBias
from optimizers import ScipyOptimizer
import os
import pickle
import sys

__author__ = "Simon Nirenberg"
__email__ = "simon_nirenberg@brown.edu"
    
class CurvatureAdaptiveABC:
    """
    Autonomous Basin Climbing (ABC) algorithm with curvature-adaptive perturbation 
    and biasing strategies.
    """

    def __init__(
        self,
        potential,
        # Optional parameters organized by category
        # Setup parameters
        starting_position=None,
        dump_every = 500, # Iters between data dumps
        dump_folder = "abc_data_dumps", 

        # Curvature estimation
        curvature_method="finite_diff",
        
        # Perturbation strategy
        perturb_type="adaptive",
        scale_perturb_by_curvature=True,
        default_perturbation_size=0.05,
        random_perturb_every = 8, 
        curvature_perturbation_scale = 1.0,
        
        # Biasing strategy
        bias_height_type ="adaptive",
        bias_covariance_type = "adaptive",
        default_bias_height = 1.0,
        min_bias_height = None,
        max_bias_height = None,
        default_bias_covariance=1.0,
        min_bias_covariance = None,
        max_bias_covariance = None,
        curvature_bias_height_scale=1.0, 
        curvature_bias_covariance_scale=1.0, 
        
        # Descent and optimization
        descent_convergence_threshold=1e-5,
        max_descent_steps=20,
        max_descent_step_size=1.0,
        max_acceptable_force_mag = 1e99 # bfgs actually does best with uncapped max force
    ):
        """
        Initialize the SmartABC sampler.
        
        Args:
            potential: Potential energy surface to sample
            expected_barrier_height: Estimated average barrier height (for scaling)
            
            See README.md for full documentation of optional parameters.
        """
        self.potential = potential
        
        # Set up configuration parameters
        self.curvature_method = curvature_method

        self.dump_every = dump_every
        self.dump_folder = dump_folder 

         # Initialize state variables
        self.reset(starting_position)
        
        self.perturb_type = perturb_type
        self.scale_perturb_by_curvature = scale_perturb_by_curvature
        self.default_perturbation_size = default_perturbation_size
        self.random_perturb_every = random_perturb_every    
        self.curvature_perturbation_scale = curvature_perturbation_scale
    
        self.bias_height_type = bias_height_type
        self.default_bias_height = default_bias_height 
        self.min_bias_height = min_bias_height
        self.max_bias_height = max_bias_height
        self.curvature_bias_height_scale = curvature_bias_height_scale

        
        self.bias_covariance_type = bias_covariance_type
        if np.isscalar(default_bias_covariance): 
            default_bias_covariance = np.eye(self.dimension) * default_bias_covariance
        self.default_bias_covariance = np.atleast_2d(default_bias_covariance)
        self.min_bias_covariance = min_bias_covariance
        self.max_bias_covariance = max_bias_covariance
        self.curvature_bias_covariance_scale = curvature_bias_covariance_scale
        
        self.descent_convergence_threshold = descent_convergence_threshold
        self.max_descent_steps = max_descent_steps
        self.max_descent_step_size = max_descent_step_size
        self.max_acceptable_force_mag = max_acceptable_force_mag
        self.potential.max_acceptable_force_mag = max_acceptable_force_mag
        self.optimizer = None

    def reset(self, starting_position=None, clean_dir=False):
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

        self.dimension = len(self.position)
        self.most_recent_hessian = None 

        self.current_iteration = 0
        self.iter_periods = [] 

   
    def store_to_disk(self):
        """Store relevant data to disk using pickle."""

        if not os.path.exists(self.dump_folder):
            os.makedirs(self.dump_folder)
        data = {
            'bias_list': deepcopy(self.bias_list),
            'unbiased_energies': deepcopy(self.unbiased_energies),
            'biased_energies': deepcopy(self.biased_energies),
            'unbiased_forces': deepcopy(self.unbiased_forces),
            'biased_forces': deepcopy(self.biased_forces),
            'trajectory': deepcopy(self.trajectory),
            'iter_periods': deepcopy(self.iter_periods),
        }
        filename = os.path.join(self.dump_folder, f"abc_dump_iter_{self.current_iteration}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Data stored to {filename}")

    @classmethod
    def load_from_disk(cls, *args, folder_path="abc_data_dumps", **kwargs):
        """Loads ABC instance from disk. Passes ABC creation args and kwargs to __init__."""

        # Find the latest dump file
        files = [f for f in os.listdir(folder_path) if f.startswith("abc_dump_iter_") and f.endswith(".pkl")]
        if not files:
            raise FileNotFoundError("No ABC dump files found in the specified folder.")
        latest_file = max(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        latest_iter = int(latest_file.split("_")[-1].split(".")[0])
        with open(os.path.join(folder_path, latest_file), "rb") as f:
            data = pickle.load(f)

        abc = cls(*args, **kwargs)
        abc.bias_list = data['bias_list']
        abc.unbiased_energies = data['unbiased_energies']
        abc.biased_energies = data['biased_energies']
        abc.unbiased_forces = data['unbiased_forces']
        abc.biased_forces = data['biased_forces']
        abc.trajectory = data['trajectory']
        abc.current_iteration = latest_iter
        abc.position = abc.trajectory[-1].copy()
        print(f"Loaded ABC state from {latest_file}")
        return abc

    def update_records(self):
        """Update iteration number and dump history if it is time to do so"""
        period = len(self.trajectory) - np.sum(self.iter_periods)
        self.iter_periods.append(period)

        if self.current_iteration % self.dump_every == 0 and self.current_iteration != 0:
            self.store_to_disk()
        self.current_iteration += 1

    # Core functionality (adapted from TraditionalABC with enhancements)
    
    def compute_biased_potential(self, position, unbiased: float = None) -> float:
        """Fully vectorized potential computation with empty bias list handling."""

        V = self.potential.potential(position) if unbiased is None else unbiased
        
        # Early return if no biases
        if not self.bias_list:
            return V
        
        # Vectorized bias potential calculation
        centers = np.array([b.center for b in self.bias_list])
        covs = np.stack([np.atleast_2d(b.covariance) for b in self.bias_list])  # Ensure each covariance is 2D
        heights = np.array([b.height for b in self.bias_list])
        
        position = np.atleast_1d(position).reshape(1, -1)  # shape (1, d)
        diffs = position - centers
        inv_covs = np.linalg.inv(covs)

        exponents = -0.5 * np.einsum('ni,nij,nj->n', diffs, inv_covs, diffs)
        V += np.sum(heights * np.exp(exponents))
        
        return V

    def compute_biased_force(self, position, unbiased: np.ndarray = None, eps=1e-5) -> np.ndarray:
        """Fully vectorized force computation with empty bias list handling."""

        if unbiased is None:
            try:
                unbiased = -self.potential.gradient(position)
            except NotImplementedError:
                # Optimized finite difference
                force = np.zeros_like(position)
                pos = position.copy()
                eps_vec = np.zeros_like(position)
                
                for i in range(len(position)):
                    eps_vec[i] = eps
                    V_plus = self.compute_biased_potential(pos + eps_vec)
                    V_minus = self.compute_biased_potential(pos - eps_vec)
                    eps_vec[i] = 0
                    force[i] = -(V_plus - V_minus) / (2 * eps)
                
                if (norm := np.linalg.norm(force)) > self.max_acceptable_force_mag:
                    force = self.max_acceptable_force_mag * force / norm
                return np.array(force)
        
        # Early return if no biases
        if not self.bias_list:
            return unbiased.copy()
        
        # Vectorized bias gradient calculation
        centers = np.array([b.center for b in self.bias_list])
        covs = np.stack([np.atleast_2d(b.covariance) for b in self.bias_list])  # Ensure each covariance is 2D
        heights = np.array([b.height for b in self.bias_list])
        
        position = np.atleast_1d(position).reshape(1, -1)  # shape (1, d)
        diffs = position - centers                      # shape (N, D)
        inv_covs = np.linalg.inv(covs)                 # shape (N, D, D)
        
        exponents = -0.5 * np.einsum('ni,nij,nj->n', diffs, inv_covs, diffs)
        mv_products = np.einsum('nij,nj->ni', inv_covs, diffs)  # shape (N, D)
        grads = -mv_products * (heights * np.exp(exponents))[:, None]  # shape (N, D)
        total_grad = -np.array(unbiased.copy()) + np.sum(grads, axis=0)
        
        force = -total_grad
        if (norm := np.linalg.norm(force)) > self.max_acceptable_force_mag:
            print(f"Warning: Clipping force magnitude from {norm:.1f} to {self.max_acceptable_force_mag}")
            force = self.max_acceptable_force_mag * force / norm
        
        return force
    
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
    
    def deposit_bias(self, center: np.ndarray = None, covariance: float|np.ndarray = None, height: float = None, verbose=False):
        """Deposit a new Gaussian bias potential.

        Args:
            center: Center of the bias (defaults to current position).
            covariance: Covariance (float for isotropic, or np.ndarray for anisotropic).
            height: Height of the bias.
        """
        pos = center if center is not None else self.position.copy()
            
        # if verbose:
        #     print('Bias position:', pos)

        if height is not None: 
            h = height 
        else: 
            if self.bias_height_type == "adaptive":
                _, curvature = self.get_softest_hessian_mode(center)
                # print("softest curvature:", curvature)
                h = self.curvature_bias_height_scale * np.abs(curvature)
            else: 
                # Fall back to defaults
                h = self.default_bias_height

        min_height = self.min_bias_height # or self.default_bias_height/2
        max_height = self.max_bias_height # or self.default_bias_height*2
        clipped_h = np.clip(h, min_height, max_height)

        if verbose and self.bias_height_type.lower() == "adaptive":
            print("Original height:", h)
            print("Clipped height:", clipped_h)

        if covariance is not None:
            cov = covariance
        else: 
            # Smart bias scaling based on curvature if available
            if self.bias_covariance_type == "adaptive":
                    # Scale covariance by inverse curvature
                    if self.most_recent_hessian is not None  and not np.any(np.isnan(self.most_recent_hessian)): 
                        cov = self.curvature_bias_covariance_scale * np.linalg.inv(
                            self.make_positive_definite(self.most_recent_hessian)
                        )
                    else:
                        raise RuntimeError("Adaptive mode selected, but Hessian is none or nan")
            else:
                cov = self.default_bias_covariance

        # 1. Get eigenvalues/vectors - ensure proper matrix shapes
        eigvals, eigvecs = np.linalg.eigh(cov)  # Use built-in SVD for stability

        # 2. Clip eigenvalues properly
        min_variance = self.min_bias_covariance # or np.min(np.diag(self.default_bias_covariance)) / 2
        max_variance = self.max_bias_covariance # or np.max(np.diag(self.default_bias_covariance)) * 2
        clipped_eigvals = np.clip(eigvals, min_variance, max_variance)


        # 3. Reconstruct with dimension guarantees
        n = len(eigvals)
        if verbose and self.bias_covariance_type.lower() =="adaptive":
            n_clipped = np.sum((eigvals != clipped_eigvals))
            print(f"Eigenvalues clipped: {n_clipped} / {n}")
            print("Original eigvals:", eigvals)
            print("Clipped eigvals:", clipped_eigvals)   

        clipped_cov = eigvecs @ np.diag(clipped_eigvals) @ eigvecs.T

        # 4. Force symmetry (numerical stability)
        clipped_cov = 0.5 * (clipped_cov + clipped_cov.T)

        try: 
            # Verification
            assert clipped_cov.shape == (n, n), "Matrix not square"
            assert np.allclose(clipped_cov, clipped_cov.T), "Not symmetric"  
        except AssertionError as e:
            print("Error:", e)
            print("Covariance matrix that caused error:\n", clipped_cov)
            sys.exit(1)

        bias = GaussianBias(
            center=pos,
            covariance=clipped_cov,
            height=clipped_h
        )
        self.bias_list.append(bias)

        if verbose and self.bias_covariance_type.lower() == "adaptive":
            approx_vol = np.sqrt(np.pow(2*np.pi, n)*np.linalg.det(clipped_cov))
            print("Approximate bias volume:", approx_vol)

    def descend(self, max_steps=None, convergence_threshold=None):
        """
        Efficient descent with built-in state recording that avoids callbacks.
        All force and energy calculations are done exactly once per step.
        """
        max_steps = max_steps or self.max_descent_steps
        convergence_threshold = convergence_threshold or self.descent_convergence_threshold
        
        max_attempts = 10
        attempt = 0
        while attempt == 0 or attempt < max_attempts and message == "Desired error not necessarily achieved due to precision loss.": 
            result, traj_data = self.optimizer.descend(self.position, max_steps=max_steps, convergence_threshold=convergence_threshold)
            final_pos = result['x']
            converged = result['converged']
            hess_inv = result['hess_inv'] if 'hess_inv' in result else None
            traj = traj_data['trajectory']
            unbiased_e = traj_data['unbiased_energies']
            biased_e = traj_data['biased_energies']
            unbiased_f = traj_data['unbiased_forces']
            biased_f = traj_data['biased_forces']
            self.trajectory.extend(traj)
            self.unbiased_energies.extend(unbiased_e)
            self.biased_energies.extend(biased_e)
            self.unbiased_forces.extend(unbiased_f)
            self.biased_forces.extend(biased_f)

            self.position = final_pos.copy()
            if 'message' in result: 
                message = result['message']
            else: 
                message = ''
            # print('attempt:', attempt)
            # print('force:', self.biased_forces[-1])
            attempt += 1 
        
        if not converged:
                print(result['message'] if 'message' in result else None)

        # Process curvature and check minimum (same as before)
        # print(hess_inv)
        self._process_curvature_info(final_pos, hess_inv)

        # check_min = check_min and not (np.all(np.isclose(self.position, self.trajectory[0], 3)))

        self._check_minimum(converged, final_pos)

        return converged

    # Curvature util

    def _process_curvature_info(self, final_pos, hess_inv=None):
        """Process curvature information from optimization result."""
        if self.curvature_method == "finite_diff":
            self.most_recent_hessian = self.compute_hessian_finite_difference(
                final_pos, 
                f0_already_computed=True
            )
        elif self.curvature_method.lower() == "bfgs" and hess_inv is not None:
            self.most_recent_hessian = np.linalg.inv(hess_inv)

    def _check_minimum(self, converged, final_pos):
        """Check if the final position is a minimum."""
        if converged:
            if self.unbiased_forces[-1] is None:
                unbiased_force = self.biased_forces[-1].copy()
                for bias in self.bias_list:
                    unbiased_force -= (-bias.gradient(final_pos))
            else:
                unbiased_force = self.unbiased_forces[-1]

            # if np.linalg.norm(unbiased_force) < threshold:
            #     # if self.most_recent_inv_hessian is not None: 
            #     #     try:
            #     #         unbiased_pes_hessian = np.linalg.inv(self.most_recent_inv_hessian)
            #     #     except np.linalg.LinAlgError:
            #     #         print("Warning: Could not invert inverse Hessian - skipping curvature analysis.")
            #     #         unbiased_pes_hessian = None
            #     # else: 
            #     #     unbiased_pes_hessian = None

            #     # if unbiased_pes_hessian is not None: 
            #     #     for bias in self.bias_list:
            #     #         unbiased_pes_hessian -= bias.hessian(final_pos)
            #     #     crit_type = self.evaluate_critical_point(unbiased_pes_hessian)
            #     #     if crit_type == "minimum": 
            #     #         self.minima.append(final_pos.copy())
            #     #     elif crit_type == "saddle":
            #     #         self.saddles.append(final_pos.copy())
            #     # else:
            #     self.minima.append(final_pos.copy())

        if np.isclose(self.unbiased_energies[-1], self.biased_energies[-1], atol=self.default_bias_height/100):
            pos = final_pos.copy()
            # check if RMSD < threshold 
            diff_threshold = 1

            def rmsd(struc, ref):
                diff = self.kabsch(struc, ref).reshape(-1, 3) - ref.reshape(-1, 3)
                return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

            if self.dimension % 3 != 0 or not any([rmsd(pos, minim) < diff_threshold for minim in self.minima]):
                self.minima.append(final_pos.copy())
                

    
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
        
    def get_softest_hessian_mode(self, position):
        """
        Gets the softest hessian mode and eigenvalue with maximal efficiency,
        accounting for the chosen curvature_method 
        """
        if self.most_recent_hessian is not None:
            hessian = self.most_recent_hessian
            return self.compute_exact_extreme_hessian_mode(hessian, desired_mode="softest")
        else: 
            raise RuntimeError("Hessian Unavailable")
       

    def make_positive_definite(self, H, eps=1e-6):
        """Modify Hessian to ensure positive definiteness."""
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals_mod = np.clip(eigvals, eps, None)
        return eigvecs @ np.diag(eigvals_mod) @ eigvecs.T

    def perturb(self, type="adaptive"):
        if type == "adaptive":
            direction, curvature = self.get_softest_hessian_mode(self.position)
            print(direction, curvature)
            if self.scale_perturb_by_curvature:
                ### CRUCIAL CODE 
                scale = np.clip(self.curvature_perturbation_scale/np.sqrt(np.abs(curvature)), self.default_perturbation_size/10, self.default_perturbation_size*10)
                ###
            else: 
                scale = self.default_perturbation_size                    
        else: 
            direction = np.random.rand(self.dimension)*2-1
            scale = self.default_perturbation_size 

        direction = direction / np.linalg.norm(direction)
        self.position += scale * direction
    
    def run(self, optimizer=None, max_iterations=100, verbose=True):
        """
        Run the ABC simulation.
        
        Args:
            max_iterations: Maximum number of iterations
            verbose: Whether to print progress
            kwargs: Override any default parameters
        """
        
        if optimizer is None:
            optimizer = ScipyOptimizer(self)
        self.optimizer = optimizer
        
        for iteration in range(self.current_iteration, max_iterations):
            # Descent phase
            # print("Before:", self.position)
            old_pos = self.position.copy()
            converged = self.descend()
            # print("delta:", delta:= np.linalg.norm(self.position-old_pos))
            # if delta < 1e-6:
            #     print("Did not sufficiently move:")
            #     print(self.biased_forces[-1])
            #     self.descend(convergence_threshold=1e-6, max_steps=1e4)

            pos = self.position.copy() # position before perturbation
           
            if self.current_iteration % self.random_perturb_every == 0: 
                self.perturb(type="random")
            else:
                self.perturb(type=self.perturb_type)


            # if self.current_iteration!=0: 
            self.deposit_bias(pos, verbose=verbose)
                
            self.update_records()

            print("hessian:", self.most_recent_hessian)
                
            if verbose:
                print(f"Iteration {iteration+1}/{max_iterations}: "
                    f"Descent converged: {converged}, "
                    f"Position:{self.position}, "
                    f"Biased Energy: {self.biased_energies[-1]}")
                print()
        
        print(f"Simulation completed.\n")
        try: 
            self.summarize()
        except Exception as e:
            print(f"Unable to complete summary:\n{e}")
    
    def summarize(self):
        """Print a comprehensive summary of the optimization results."""
        output = []
        
        # 1. Report all found minima with energies
        if self.minima:
            output.append("\nEnergies of found minima:")
            for i, min_pos in enumerate(self.minima):
                idx = np.argmin(np.linalg.norm(np.array(self.trajectory) - np.array(min_pos), axis=1))
                output.append(f"Minimum {i+1}: Position = {min_pos}, Energy = {self.unbiased_energies[idx]}")

        # 2. Report any on-the-fly saddle points
        if self.saddles: 
            output.append(f"\nOn-the-fly-identified saddle-points: {self.saddles}")

        # 3. Calculate approximate saddle points between minima
        if len(self.minima) > 1:
            minima_indices = sorted([
                np.argmin(np.linalg.norm(np.array(self.trajectory) - np.array(min_pos), axis=1))
                for min_pos in self.minima
            ])
            
            saddle_info = []
            for i in range(len(minima_indices) - 1):
                start, end = minima_indices[i], minima_indices[i + 1]
                segment = slice(start, end + 1)
                max_idx = np.argmax(self.unbiased_energies[segment])
                
                saddle_info.append((
                    self.trajectory[start + max_idx],
                    self.unbiased_energies[start + max_idx]
                ))
            
            output.append("\nApproximate saddle points between minima:")
            for i, (pos, energy) in enumerate(saddle_info):
                output.append(f"Saddle {i+1}: Position = {pos}, Energy = {energy}")
                self.saddles.append(pos)

        # 4. Report computational statistics
        output.append("\nComputational statistics:")
        output.append(f"\tEnergy evaluations: {self.potential.energy_calls}")
        output.append(f"\tForce evaluations: {self.potential.force_calls}")
        output.append(f"\tTotal steps: {len(self.trajectory)}")

        # 5. Report verification against known minima (if available) - MOVED TO END
        if hasattr(self.potential, "known_minima"):
            true_minima = self.potential.known_minima()
            found_minima = np.array(self.minima)
            matched = [np.any(np.linalg.norm(found_minima - np.array(true_min), axis=1) < 1e-3) for true_min in true_minima]
            
            output.append("\nVerification against reference:")
            output.append(f"True minima found: {sum(matched)}/{len(true_minima)}")
            if not all(matched):
                output.append("Missed minima at positions:")
            for i, found in enumerate(matched):
                if not found:
                    output.append(str(true_minima[i]))

        if hasattr(self.potential, "known_saddles"):
            true_saddles = self.potential.known_saddles()
            found_saddles = np.array(self.saddles)
            
            # Only proceed if we have found saddles and they exist
            if len(found_saddles) > 0:
                # Ensure found_saddles is 2D (reshape if necessary)
                if found_saddles.ndim == 1:
                    found_saddles = found_saddles.reshape(-1, 1)
                    
                matched_saddles = [np.any(np.linalg.norm(found_saddles - true_sad.reshape(1, -1), axis=1) < 1e-2) 
                                for true_sad in true_saddles]
            else:
                matched_saddles = [False] * len(true_saddles)

            output.append("\nVerification against reference (saddles):")
            output.append(f"True saddles found: {sum(matched_saddles)}/{len(true_saddles)}")
            if not all(matched_saddles):
                output.append("Missed saddles at positions:")
                for i, found in enumerate(matched_saddles):
                    if not found:
                        output.append(str(true_saddles[i]))

        # Print all collected output
        print('\n'.join(output))