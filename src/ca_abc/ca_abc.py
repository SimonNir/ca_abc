from copy import deepcopy
import numpy as np
from ca_abc.bias import GaussianBias
from ca_abc.optimizers import FIREOptimizer
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
        # Note dumping can be very storage-intensive, especially with high dimensionality
        dump_every = 10000, # Iters between data dumps
        dump_folder = "abc_data_dumps", 

        # Curvature estimation
        curvature_method="bfgs", # 'finite_diff', 'bfgs', or 'lanczos' (experimental)
        
        # Perturbation strategy
        perturb_type="adaptive",  # None or "none" - Kushima et al. style deterministic exploration; no perturbation
        # "adaptive" - deterministic softmode perturbation
        # "stochastic" - random perturbation, allowing parallel exploration
        # "adaptive_stochastic" - randomized softmode perturbation (randomly selects +v or -v for softest mode v)
        scale_perturb_by_curvature=True,
        default_perturbation_size=0.05,
        min_perturbation_size = None, 
        max_perturbation_size = None, 
        curvature_perturbation_scale = 1.0, # only used if ema scaling is false 
        
        # Biasing strategy
        bias_height_type ="adaptive",
        default_bias_height = 1.0,
        min_bias_height = None,
        max_bias_height = None,
        curvature_bias_height_scale=1.0, # only used if ema scaling is false 

        bias_covariance_type = "adaptive",
        default_bias_covariance=1.0,
        min_bias_covariance = None,
        max_bias_covariance = None,
        curvature_bias_covariance_scale=1.0, # only used if ema scaling is false 

 
        use_ema_adaptive_scaling = True, 
        ema_alpha = 0.3, 
        conservative_ema_delta = False,

        energy_diff_threshold = None, # energy difference threshold at which significance is considered (e.g. if diff < threshold after relaxation, considered new minimum)
        struc_uniqueness_rmsd_threshold = 1e-3,
        bias_filtering_cutoff = 1000, # bias_list length after which we start ignoring biases further than 3.5 sigma_max from the current position, for speedups
        
        
        # Descent and optimization
        descent_convergence_threshold=1e-4,
        min_descent_steps=5,
        max_descent_steps=600,
        max_acceptable_force_mag = 1e99, # in practice, the optimizers actually do best with uncapped max force

        biased_atom_indices = None, # For atomistic simulations, specifies which atoms to apply bias to; by default (None), applies to all atoms
    ):
        """Initializes the Curvature-Adaptive Autonomous Basin Climbing algorithm.

        This class implements the ABC algorithm to explore potential energy
        surfaces, using local curvature to adaptively modify the biasing and
        perturbation strategies for efficient landscape exploration.

        Args:
            potential (object): The potential energy surface to explore. Must
                provide `potential()` and `gradient()` methods.

            ## Setup Parameters
            starting_position (np.ndarray, optional): Initial coordinates for the
                simulation. If None, uses the potential's default start.
            dump_every (int, optional): Frequency (in iterations) for saving
                simulation data to disk.
            dump_folder (str, optional): Directory to save data dumps.

            ## Curvature Estimation
            curvature_method (str, optional): Method to estimate the Hessian.
                Options: 'bfgs', 'finite_diff', 'lanczos'. Defaults to 'bfgs'.

            ## Perturbation Strategy
            perturb_type (str, optional): Strategy for perturbing the system
                after a minimum is found. Defaults to "adaptive".
                - "none": No perturbation is applied (deterministic non-adaptive).
                - "stochastic": Perturbs along a fully random direction.
                - "adaptive_stochastic": Perturbs along the softest Hessian
                  mode with a randomly chosen sign (+v or -v).
                - "adaptive": Deterministically perturbs along the softest mode,
                  aligning it with the last displacement vector for consistency.
            scale_perturb_by_curvature (bool, optional): If True, scales the
                perturbation size based on the softest mode's curvature.
            default_perturbation_size (float, optional): Default perturbation
                magnitude if not scaling by curvature.
            min_perturbation_size (float, optional): Minimum allowed perturbation.
            max_perturbation_size (float, optional): Maximum allowed perturbation.
            curvature_perturbation_scale (float, optional): Scaling factor for
                curvature-based perturbation size if EMA is disabled.

            ## Biasing Strategy
            bias_height_type (str, optional): Method for determining bias height.
                'adaptive' uses curvature, otherwise uses default.
            default_bias_height (float, optional): Default height for Gaussian bias.
            min_bias_height (float, optional): Minimum allowed bias height.
            max_bias_height (float, optional): Maximum allowed bias height.
            curvature_bias_height_scale (float, optional): Scaling factor for
                curvature-based bias height if EMA is disabled.
            bias_covariance_type (str, optional): Method for determining bias
                covariance. 'adaptive' uses the inverse Hessian.
            default_bias_covariance (float or np.ndarray, optional): Default
                covariance for Gaussian bias.
            min_bias_covariance (float, optional): Minimum allowed variance along
                any principal axis of the bias.
            max_bias_covariance (float, optional): Maximum allowed variance.
            curvature_bias_covariance_scale (float, optional): Scaling factor
                for curvature-based covariance if EMA is disabled.

            ## Adaptive Scaling
            use_ema_adaptive_scaling (bool, optional): If True, uses an
                Exponential Moving Average to smooth adaptive parameters over time.
            ema_alpha (float, optional): Smoothing factor for EMA (0 < alpha <= 1).
            conservative_ema_delta (bool, optional): If True, prevents the EMA
                logic from excessively widening the bias covariance.

            ## Convergence and Uniqueness
            energy_diff_threshold (float, optional): Energy threshold to consider a
                point a new, unbiased minimum.
            struc_uniqueness_rmsd_threshold (float, optional): RMSD threshold
                to determine if a found minimum is unique.
            bias_filtering_cutoff (int, optional): Number of biases after which
                to start spatial filtering for performance.

            ## Descent and Optimization
            descent_convergence_threshold (float, optional): Force threshold for
                optimizer convergence.
            min_descent_steps (int, optional): Minimum number of steps the
                optimizer must take, even if converged.
            max_descent_steps (int, optional): Maximum number of steps for the
                optimizer per iteration.
            max_acceptable_force_mag (float, optional): A cap for clipping
                excessively large forces.

            ## Subspace Biasing
            biased_atom_indices (list[int], optional): If provided, applies biases
                and perturbations only to the specified atom indices. Defaults
                to all atoms.
        """

        self.potential = potential
        
        # Set up configuration parameters
        self.curvature_method = curvature_method

        self.dump_every = dump_every
        self.dump_folder = dump_folder 

         # Initialize state variables
        self.reset(starting_position)

        if str(perturb_type).lower() in ['adaptive', 'none', 'stochastic', 'adaptive_stochastic']:
            self.perturb_type = perturb_type
        else:
            raise ValueError("Invalid perturbation type specified")


        self.scale_perturb_by_curvature = scale_perturb_by_curvature
        self.default_perturbation_size = default_perturbation_size
        if min_perturbation_size is None: 
            min_perturbation_size = default_perturbation_size
        self.min_perturbation_size = min_perturbation_size
        self.max_perturbation_size = max_perturbation_size
        self.curvature_perturbation_scale = curvature_perturbation_scale  # only used if ema scaling is false 
    
        self.bias_height_type = bias_height_type
        self.default_bias_height = default_bias_height 
        if min_bias_height is None:
            min_bias_height = default_bias_height
        self.min_bias_height = min_bias_height
        if max_bias_height is None:
            max_bias_height = default_bias_height
        self.max_bias_height = max_bias_height
        self.curvature_bias_height_scale = curvature_bias_height_scale  # only used if ema scaling is false 

        
        self.bias_covariance_type = bias_covariance_type
        self.default_bias_covariance = default_bias_covariance
        if min_bias_covariance is None:
            min_bias_covariance = self.default_bias_covariance if np.isscalar(self.default_bias_covariance) else np.min(np.diag(self.default_bias_covariance))
        self.min_bias_covariance = min_bias_covariance
        if max_bias_covariance is None:
            max_bias_covariance = self.default_bias_covariance if np.isscalar(self.default_bias_covariance) else np.max(np.diag(self.default_bias_covariance))
        self.max_bias_covariance = max_bias_covariance
        self.curvature_bias_covariance_scale = curvature_bias_covariance_scale # only used if ema scaling is false 

        self.use_ema_adaptive_scaling = use_ema_adaptive_scaling
        self.ema_alpha = ema_alpha
        self.conservative_ema_delta = conservative_ema_delta

        self.log_running_variance_ema = None
        self.log_running_height_ema = None
        self.log_running_perturb_ema = None

        self.energy_diff_threshold = energy_diff_threshold if energy_diff_threshold is not None else self.min_bias_height/50
        self.struc_uniqueness_rmsd_threshold = struc_uniqueness_rmsd_threshold

        self.bias_filtering_cutoff = bias_filtering_cutoff
        
        self.descent_convergence_threshold = descent_convergence_threshold
        self.min_descent_steps = min_descent_steps
        if (self.perturb_type is None or self.perturb_type == "None") and self.min_descent_steps is None:
            print('Warning: You have chosen no perturbation but also not specified a min_descent_steps; \nthis may result in overbiasing.Consider setting min_descent_steps > 0.') 
        self.max_descent_steps = max_descent_steps
        self.max_acceptable_force_mag = max_acceptable_force_mag
        self.potential.max_acceptable_force_mag = max_acceptable_force_mag
        self.optimizer = None

        if self.curvature_method not in ["bfgs", "finite_diff", "lanczos"] and "adaptive" in [self.bias_covariance_type, self.bias_height_type, self.perturb_type]:
            print("Warning: adaptive mode chosen for one or more parameters, but no valid curvature method supplied. Setting to \'bfgs\' by default.")
            self.curvature_method = "bfgs"

        self.biased_atom_indices = biased_atom_indices
         # Construct the DOF indices once and store them as an attribute.
        if self.biased_atom_indices is not None:
            dof_indices = []
            for i in self.biased_atom_indices:
                dof_indices.extend([3 * i, 3 * i + 1, 3 * i + 2])
            self.dof_indices = np.array(dof_indices)
        else:
            self.dof_indices = None # Signal that biases are full-dimensional


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
        self.min_indices = []
        self.saddles = []
        self.saddle_indices = []

        self.dimension = len(self.position)
        self.most_recent_hessian = None 

        self.current_iteration = 0
        self.iter_periods = [] 
        self.energy_calls_at_each_min = []
        self.force_calls_at_each_min = []
   
    def store_to_disk(self, remove_prior = False):
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
            'energy_calls_at_each_min': deepcopy(self.energy_calls_at_each_min),
            'force_calls_at_each_min': deepcopy(self.force_calls_at_each_min)
        }
        filename = os.path.join(self.dump_folder, f"abc_dump_iter_{self.current_iteration}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Data stored to {filename}")

        old_iter = self.current_iteration - self.dump_every
        if remove_prior and old_iter >= 0:
            old_name = os.path.join(self.dump_folder, f"abc_dump_iter_{old_iter}.pkl")
            if os.path.exists(old_name):
                try:
                    os.remove(old_name)
                    print(f"Deleted previous dump at {old_name}.")
                except OSError as e:
                    print(f"Warning: could not delete {old_name}: {e}")

    @classmethod
    def load_from_disk(cls, *args, folder_path="abc_data_dumps", minimal=True, **kwargs):
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
        if not minimal:
            abc.unbiased_energies = data['unbiased_energies']
            abc.biased_energies = data['biased_energies']
            abc.unbiased_forces = data['unbiased_forces']
            abc.biased_forces = data['biased_forces']
            abc.trajectory = data['trajectory']
            abc.energy_calls_at_each_min = data['energy_calls_at_each_min']
            abc.force_calls_at_each_min = data['force_calls_at_each_min']
        abc.current_iteration = latest_iter
        abc.position = data['trajectory'][-1].copy()
        print(f"Loaded ABC state from {latest_file}")
        if minimal:
            print("Used minimal loading: \nyou will need to load and analyze each .pkl separately, " \
            "as the bias list is the only carry-over into local memory.\nMake sure remove_prior=False "
            "in future calls to store_to_disk.")
        return abc

    def update_records(self):
        """Update iteration number and dump history if it is time to do so"""
        period = len(self.trajectory) - np.sum(self.iter_periods)
        self.iter_periods.append(period)

        if self.current_iteration % self.dump_every == 0 and self.current_iteration != 0:
            self.store_to_disk()
        self.current_iteration += 1

    # Core functionality (adapted from TraditionalABC with enhancements)

    def update_bias_cache(self):
        """
        Call this to rebuild cached vectorized arrays when bias_list changes.
        Reconstructs full inverse covariance matrices from reduced ones if necessary.
        """
        if not self.bias_list:
            self._bias_centers = np.empty((0, self.position.size))
            self._bias_heights = np.array([])
            self._bias_inv_covs = np.empty((0, self.position.size, self.position.size))
        else:
            self._bias_centers = np.array([b.center for b in self.bias_list])
            self._bias_heights = np.array([b.height for b in self.bias_list])
            
            # Reconstruct full inverse covariance matrices for the vectorized calculation
            inv_covs_list = []
            for b in self.bias_list:
                if b.dof_indices is None:
                    # This is a full-dimensional bias
                    inv_covs_list.append(b._cov_inv)
                else:
                    # This is a reduced bias; build the full sparse matrix
                    full_inv_cov = np.zeros((self.dimension, self.dimension))
                    full_inv_cov[np.ix_(b.dof_indices, b.dof_indices)] = b._cov_inv
                    inv_covs_list.append(full_inv_cov)
            
            self._bias_inv_covs = np.stack(inv_covs_list)
            
        self._bias_cache_valid = True 

    def compute_biased_potential(self, position, unbiased: float = None) -> float:
        V = self.potential.potential(position) if unbiased is None else unbiased

        if not self.bias_list:
            return V

        if not getattr(self, "_bias_cache_valid", False):
            self.update_bias_cache()

        pos = np.atleast_2d(position).reshape(1, -1)  # (1, d)
        diffs = pos - self._bias_centers              # (N_bias, d)

        if len(self.bias_list) > self.bias_filtering_cutoff:
            # Apply fast Euclidean filtering
            sigma_max = np.sqrt(self.max_bias_covariance)
            threshold_sq = (3.5 * sigma_max) ** 2
            distsq = np.sum(diffs**2, axis=1)
            mask = distsq < threshold_sq

            if not np.any(mask):
                return V  # No nearby biases contribute

            # Filter arrays
            diffs = diffs[mask]
            inv_covs = self._bias_inv_covs[mask]
            heights = self._bias_heights[mask]

            exponents = -0.5 * np.einsum('ni,nij,nj->n', diffs, inv_covs, diffs)
            V += np.sum(heights * np.exp(exponents))
        else:
            # Fast path: no filtering
            exponents = -0.5 * np.einsum('ni,nij,nj->n', diffs, self._bias_inv_covs, diffs)
            V += np.sum(self._bias_heights * np.exp(exponents))

        return V
    
    def compute_biased_force(self, position, unbiased: np.ndarray = None, eps=1e-5) -> np.ndarray:
        if unbiased is None:
            try:
                unbiased = -self.potential.gradient(position)
            except NotImplementedError:
                # Finite difference fallback
                force = np.zeros_like(position)
                pos = position.copy()
                eps_vec = np.zeros_like(position)
                for i in range(len(position)):
                    eps_vec[i] = eps
                    V_plus = self.compute_biased_potential(pos + eps_vec)
                    V_minus = self.compute_biased_potential(pos - eps_vec)
                    eps_vec[i] = 0
                    force[i] = -(V_plus - V_minus) / (2 * eps)
                norm = np.linalg.norm(force)
                if norm > self.max_acceptable_force_mag:
                    force = self.max_acceptable_force_mag * force / norm
                return force

        if not self.bias_list:
            return unbiased.copy()

        if not getattr(self, "_bias_cache_valid", False):
            self.update_bias_cache()

        pos = np.atleast_2d(position).reshape(1, -1)  # (1, d)
        diffs = pos - self._bias_centers              # (N_bias, d)

        if len(self.bias_list) > self.bias_filtering_cutoff:
            # Use Euclidean distance filtering
            sigma_max = np.sqrt(self.max_bias_covariance)
            threshold_sq = (3.5 * sigma_max) ** 2
            distsq = np.sum(diffs**2, axis=1)
            mask = distsq < threshold_sq

            if not np.any(mask):
                return unbiased.copy()  # No nearby bias contribution

            diffs = diffs[mask]
            inv_covs = self._bias_inv_covs[mask]
            heights = self._bias_heights[mask]
        else:
            inv_covs = self._bias_inv_covs
            heights = self._bias_heights

        exponents = -0.5 * np.einsum('ni,nij,nj->n', diffs, inv_covs, diffs)       # (N_bias,)
        mv_products = np.einsum('nij,nj->ni', inv_covs, diffs)                     # (N_bias, d)
        grads = -mv_products * (heights * np.exp(exponents))[:, None]             # (N_bias, d)
        total_grad = -unbiased + np.sum(grads, axis=0)                            # (d,)

        force = -total_grad
        norm = np.linalg.norm(force)
        if norm > self.max_acceptable_force_mag:
            print(f"Warning: Clipping force magnitude from {norm:.1f} to {self.max_acceptable_force_mag}")
            force = self.max_acceptable_force_mag * force / norm

        return force
    
    def compute_hessian_finite_difference(self, position, force_symmetric=True, eps=1e-3):
        """
        Compute the full Hessian matrix of the potential at `position`
        using central differences on the forces.

        Args:
            position (ndarray): shape (n,), atomic coordinates or generalized coords
            force_symmetric (bool): enforce Hessian symmetry
            eps (float): displacement step for finite difference

        Returns:
            hessian (ndarray): shape (n, n), Hessian of the potential
        """
        n = len(position)
        hessian = np.zeros((n, n))
        unit = np.eye(n)

        for j in range(n):
            pos_plus = position + eps * unit[j]
            pos_minus = position - eps * unit[j]

            f_plus = self.compute_biased_force(pos_plus)
            f_minus = self.compute_biased_force(pos_minus)

            df_dxj = (f_plus - f_minus) / (2 * eps)
            hessian[:, j] = -df_dxj  # column j

        if force_symmetric:
            hessian = 0.5 * (hessian + hessian.T)

        return hessian
    
    # Experimental! 
    def estimate_hessian_lanczos(self,
                                    position,
                                    k=6,
                                    mode='softest',        # 'softest' or 'hardest'
                                    eps=1e-4,              # step for Hv finite-difference
                                    which=None,            # internal arg derived from mode
                                    tol=1e-6,
                                    maxiter=None,
                                    random_state=None,
                                    enforce_symmetry=True):
        """
        Estimate an approximate dense Hessian using Lanczos (ARPACK via eigsh)
        and Hessian-vector products computed by finite-differencing forces.

        Args:
            self: object providing `compute_biased_force(position)` -> ndarray (n,)
            position: ndarray, shape (n,), point at which Hessian is desired
            k: int, number of eigenpairs to compute (k << n)
            mode: 'softest' -> compute k smallest eigenvalues; 'hardest' -> compute k largest
            eps: float, finite-difference step for Hessian-vector product Hv
            tol: float, eigsh tolerance
            maxiter: int or None, eigsh max iterations
            random_state: int or None, seed for ARPACK restart vector
            enforce_symmetry: bool, symmetrize final matrix

        Returns:
            H_approx: ndarray (n,n), dense approximate Hessian
        """
        from scipy.sparse.linalg import LinearOperator, eigsh
        position = np.asarray(position, dtype=float)
        n = position.size
        if k <= 0 or k > n:
            raise ValueError("k must be between 1 and n")

        # Decide which eigenvalues to request from eigsh
        if mode == 'hardest':
            which = 'LM'  # largest magnitude (largest eigenvalues if positive-definite)
        elif mode == 'softest':
            which = 'SM'  # smallest magnitude (soft modes)
        else:
            raise ValueError("mode must be 'softest' or 'hardest'")

        # Helper: Hessian-vector product via two-sided finite difference on forces.
        # Hv ≈ - (F(x + eps * v) - F(x - eps * v)) / (2 * eps)
        def Hv(v):
            # ARPACK may pass v as dtype=float64; ensure same shape
            v = np.asarray(v, dtype=float).ravel()
            # small safeguard if v is zero
            norm_v = np.linalg.norm(v)
            if norm_v == 0.0:
                return np.zeros_like(v)

            pos_plus = position + eps * v
            pos_minus = position - eps * v

            f_plus = self.compute_biased_force(pos_plus)   # shape (n,)
            f_minus = self.compute_biased_force(pos_minus)

            # Note: F = -∇V, so (F(x+)-F(x-))/(2 eps) ≈ ∇F · v = -H v
            Hv_est = - (f_plus - f_minus) / (2.0 * eps)
            return Hv_est

        # Wrap as a LinearOperator for eigsh
        linop = LinearOperator((n, n), matvec=Hv, dtype=float)

        # Call eigsh to compute k eigenpairs
        # For 'SM' sometimes ARPACK struggles; if that happens you can try shift-invert via sigma=0
        try:
            vals, vecs = eigsh(linop, k=k, which=which, tol=tol, maxiter=maxiter, v0=None, return_eigenvectors=True)
        except Exception as e:
            # Fallback: if requesting SM failed, try shift-invert around 0 (sigma=0) to obtain smallest algebraic
            if mode == 'softest':
                vals, vecs = eigsh(linop, k=k, sigma=0.0, tol=tol, maxiter=maxiter, return_eigenvectors=True)
            else:
                raise

        # Ensure real values (numerical imaginary parts should be negligible)
        vals = np.real_if_close(vals, tol=1000)
        vecs = np.real_if_close(vecs, tol=1000)

        # Sort eigenpairs by eigenvalue ascending
        idx_sort = np.argsort(vals)
        vals = vals[idx_sort]
        vecs = vecs[:, idx_sort]

        # Compute the scaling scalar mu for the identity remainder.
        # We want mu to be "intermediate" in magnitude relative to the found eigenvalues.
        # Robust strategy:
        #   - use geometric mean between max(|vals|) and min_nonzero(|vals|)
        #   - if all values are zero or tiny, fallback to 1.0
        abs_vals = np.abs(vals)
        eps_small = 1e-12
        max_abs = max(abs_vals.max(), eps_small)
        min_nonzero = max(abs_vals[abs_vals > eps_small].min() if np.any(abs_vals > eps_small) else eps_small, eps_small)

        # geometric mean of magnitudes
        mu_mag = np.sqrt(max_abs * min_nonzero)

        # choose mu with sign positive (we want identity-like positive stiffness)
        mu = float(mu_mag)

        # Build dense approximation:
        # H_approx = mu * I + Q (Lambda - mu I) Q^T
        # where Q has the k eigenvectors as columns, Lambda diag(vals)
        Q = vecs  # shape (n, k)
        Lambda = np.diag(vals)  # shape (k, k)

        # Compute Q (Lambda - mu I_k) Q^T efficiently
        # delta = (Lambda - mu I_k)
        delta = Lambda - (mu * np.eye(k))
        H_correction = Q @ delta @ Q.T

        H_approx = mu * np.eye(n) + H_correction

        if enforce_symmetry:
            # Force exact symmetry to remove numerical asymmetry
            H_approx = 0.5 * (H_approx + H_approx.T)

        return H_approx


    def deposit_bias(self, center: np.ndarray = None, covariance: float | np.ndarray = None, height: float = None, verbose=False):
        pos = center if center is not None else self.position.copy()
        dof_indices = self.dof_indices

        # --- Extract sub-Hessian ---
        if self.most_recent_hessian is not None and dof_indices is not None:
            sub_hessian = self.most_recent_hessian[np.ix_(dof_indices, dof_indices)]
        else:
            sub_hessian = self.most_recent_hessian

        # --- Determine Bias Height ---
        if height is not None:
            h = height
        else:
            if self.bias_height_type == "adaptive" and sub_hessian is not None:
                _, curvature = self.compute_exact_extreme_hessian_mode(sub_hessian, desired_mode="softest")
                if self.use_ema_adaptive_scaling:
                    log_height = np.log(self.curvature_bias_height_scale * np.abs(curvature))
                    if self.log_running_height_ema is None:
                        self.log_running_height_ema = log_height
                    else:
                        self.log_running_height_ema = (
                            self.ema_alpha * log_height +
                            (1 - self.ema_alpha) * self.log_running_height_ema
                        )
                    h = np.exp(self.log_running_height_ema)
                else:
                    h = self.curvature_bias_height_scale * np.abs(curvature)
            else:
                h = self.default_bias_height

        clipped_h = np.clip(h, self.min_bias_height, self.max_bias_height)
        if verbose and self.bias_height_type.lower() == "adaptive":
            print("Proposed height:", h)
            print("Clipped height:", clipped_h)

        # --- Construct Reduced Covariance ---
        reduced_dim = len(dof_indices) if dof_indices is not None else self.dimension   
        if covariance is not None:
            reduced_cov = covariance if not np.isscalar(covariance) else np.eye(reduced_dim) * covariance
        else:
            if self.bias_covariance_type == "adaptive" and sub_hessian is not None:
                reduced_cov = np.linalg.inv(sub_hessian)
            else:
                if np.isscalar(self.default_bias_covariance):
                    reduced_cov = np.eye(reduced_dim) * self.default_bias_covariance
                else:
                    reduced_cov = self.default_bias_covariance[np.ix_(dof_indices, dof_indices)] if reduced_dim != len(self.default_bias_covariance) else self.default_bias_covariance

        # --- Eigendecompose and Clip in Subspace ---
        eigvals, eigvecs = np.linalg.eigh(np.atleast_2d(reduced_cov))

        if self.bias_covariance_type == "adaptive":
            if self.use_ema_adaptive_scaling and sub_hessian is not None:
                proposed_vars = clipped_h * eigvals

                log_mean_var = np.mean(np.log(proposed_vars))

                if self.log_running_variance_ema is None:
                    self.log_running_variance_ema = log_mean_var
                else:
                    self.log_running_variance_ema = (
                        self.ema_alpha * log_mean_var +
                        (1 - self.ema_alpha) * self.log_running_variance_ema
                    )

                min_variance = self.min_bias_covariance
                max_variance = self.max_bias_covariance 
                log_target_var = 0.5 * (np.log(min_variance) + np.log(max_variance))

                log_delta = log_target_var - self.log_running_variance_ema
                delta = np.exp(log_delta)
                if self.conservative_ema_delta:
                    delta = np.clip(delta, 0.1, 1) # prevent overbiasing by keeping delta <= 1
                
                var_along_modes = delta * proposed_vars

                if verbose:
                    print("EMA log variance:", self.log_running_variance_ema)
                    print("Target log variance:", log_target_var)
                    print("Dynamic delta_cov:", delta)
            else:
                var_along_modes =  clipped_h * self.curvature_bias_covariance_scale * eigvals
        else:
            var_along_modes = eigvals

        clipped_eigvals = np.clip(var_along_modes, self.min_bias_covariance, self.max_bias_covariance)

        if verbose and self.bias_covariance_type.lower() == "adaptive":
            print("Proposed var:", var_along_modes)
            print("Clipped var:", clipped_eigvals)

        clipped_cov_reduced = eigvecs @ np.diag(clipped_eigvals) @ eigvecs.T
        clipped_cov_reduced = np.atleast_2d(0.5 * (clipped_cov_reduced + clipped_cov_reduced.T))

        # --- Create and Store Bias ---
        # The bias object now stores the reduced covariance and the indices
        bias = GaussianBias(center=pos, 
                            covariance=clipped_cov_reduced, 
                            height=clipped_h,
                            dof_indices=dof_indices)
        self.bias_list.append(bias)
        self._bias_cache_valid = False # Invalidate the cache


    def descend(self, verbose=False):
        """
        Efficient descent with built-in state recording that avoids callbacks.
        All force and energy calculations are done exactly once per step.
        """
        max_steps = self.max_descent_steps
        min_steps= self.min_descent_steps
        convergence_threshold = self.descent_convergence_threshold
        
        max_attempts = 5
        attempt = 0
        while attempt == 0 or attempt < max_attempts and message == "Desired error not necessarily achieved due to precision loss.": 
            result, traj_data = self.optimizer.descend(self.position, max_steps=max_steps, convergence_threshold=convergence_threshold, min_steps=min_steps, verbose=verbose)
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
                message = None
            attempt += 1 
        
        if not converged:
                if 'message' in result: 
                    print(result['message'])
        
        self._process_curvature_info(final_pos, hess_inv)

        self._check_minimum(converged, final_pos)

        return converged

    # Curvature util

    def _process_hess_inv(self, hess_inv, verbose=True):
        """
        Regularizes the inverse Hessian by:
        - symmetrizing,
        - clipping eigenvalues to [1e-6, 1e6],
        - reconstructing,
        - inverting to get Hessian.
        """
        def symmetrize(H):
            return 0.5 * (H + H.T)

        try:
            H_inv_sym = symmetrize(hess_inv)
            eigvals, eigvecs = np.linalg.eigh(H_inv_sym)

            if np.any(eigvals<=0):
                if verbose: 
                    print(f"BFGS Hessian has eigenvalue(s) <= 0 and cannot be reliably trusted; reverting to default behavior. \nEigenvalues: {eigvals}")
                self.most_recent_hessian = None 
                return 
            
            if np.abs(np.max(eigvals)/np.min(eigvals)) > 1e6: 
                if verbose: 
                    print(f"Approximate Hessian has condition number >1e6 and cannot be reliably trusted; reverting to default behavior. \nEigenvalues: {eigvals}")
                self.most_recent_hessian = None
                return 

            # Step 6: Invert to get Hessian
            self.most_recent_hessian = np.linalg.inv(H_inv_sym)

        except np.linalg.LinAlgError as e:
            if verbose:
                print(f"Failed to process inverse Hessian: {e}")
            self.most_recent_hessian = None


    def _process_curvature_info(self, final_pos, hess_inv=None, verbose=True):
        """Process curvature information from optimization result."""
        if self.curvature_method == "finite_diff":
            self.most_recent_hessian = self.compute_hessian_finite_difference(
                final_pos, 
            )
        elif self.curvature_method.lower() == "lanczos":
            self.most_recent_hessian = self.estimate_hessian_lanczos(
                final_pos,
            )
        elif self.curvature_method.lower() == "bfgs":
            if hess_inv is not None: 
                self._process_hess_inv(hess_inv, verbose=verbose)
            else: 
                self.most_recent_hessian = None

    def _check_minimum(self, converged, final_pos, verbose=True):
        """Check if the final position is a minimum."""
        if np.isclose(self.unbiased_energies[-1], self.biased_energies[-1], atol=self.energy_diff_threshold):
            
            def is_unique(pos, minima, threshold=1e-3):
                if len(minima) == 0:
                    return True

                minima_array = np.array(minima)
                diff = minima_array - pos
                rmsds = np.sqrt(np.mean(diff**2, axis=1)) 
                
                return np.all(rmsds >= threshold)
            
            if is_unique(final_pos, self.minima, threshold=self.struc_uniqueness_rmsd_threshold):
                print(f"Identified minimum at position {final_pos} with unbiased energy {self.unbiased_energies[-1]}")
                if not converged and verbose:
                    print("Warning: minimum above was identified at a position where optimizer did not converge to desired tolerance")

                # Always append the new minimum
                self.minima.append(final_pos.copy())
                min_ind = len(self.trajectory) - 1
                self.min_indices.append(min_ind)
                # Append metadata for the (still valid) new minimum
                self.energy_calls_at_each_min.append(self.potential.energy_calls)
                self.force_calls_at_each_min.append(self.potential.force_calls)

                # Only do saddle logic if we now have at least 2 minima
                if len(self.min_indices) >= 2:
                    # Compute saddle index between last two minima
                    prev_min_ind = self.min_indices[-2]
                    curr_min_ind = self.min_indices[-1]
                    start = prev_min_ind
                    end = curr_min_ind + 1  # include current minimum

                    saddle_ind = start + np.argmax(self.unbiased_energies[start:end])
                    self.saddle_indices.append(saddle_ind)
                    self.saddles.append(self.trajectory[saddle_ind])

                    # Now validate the *previous* minimum
                    prev_min_energy = self.unbiased_energies[prev_min_ind]
                    saddle_energy = self.unbiased_energies[saddle_ind]

                    if saddle_energy - prev_min_energy < self.min_bias_height:
                        print(f"Removing previous minimum at index {prev_min_ind} (energy {prev_min_energy}) because",
                              f"\ncorresponding saddle does not have sufficiently greater energy ({saddle_energy})")

                        # Remove the previous minimum and the newly added saddle
                        self.minima.pop(-2)
                        self.min_indices.pop(-2)
                        self.saddle_indices.pop()
                        self.saddles.pop()

                        # Optionally remove corresponding metadata if tracked
                        self.energy_calls_at_each_min.pop(-2)
                        self.force_calls_at_each_min.pop(-2)                
    
    def compute_exact_extreme_hessian_mode(self, hessian, desired_mode = "softest"):
        """
        Fully diagonalize the Hessian to find the exact softest mode.
        More efficient than estimator for low-dim potentials, worse for high-dim
        """
        eigvals, eigvecs = np.linalg.eigh(hessian)
        # Pair each eigenvector with its corresponding eigenvalue
        dirs_and_curvatures = [(eigvecs[:, i], eigvals[i]) for i in range(len(eigvals))]
        # Sort by curvature (eigenvalue)
        sorted_dirs = sorted(dirs_and_curvatures, key=lambda x: x[1])

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

    def perturb(self, type="adaptive", verbose=False):
        """
        Applies a perturbation to self.position based on the specified type.
        Falls back to 'none' for adaptive and 'stochastic' for adaptive_stochastic if Hessian is missing.
        """
        type_lower = None if type is None else str(type).lower()

        # Mode 1: No perturbation
        if type_lower in [None, "none"]:
            if verbose:
                print("Perturbation type is 'none' or None. Skipping perturbation.")
            return

        # Mode 2: Fully random (stochastic) perturbation
        if type_lower == "stochastic":
            direction = np.random.randn(self.dimension)
            scale = self.default_perturbation_size
            self.position += scale * (direction / np.linalg.norm(direction))
            if verbose:
                print(f"Applied stochastic perturbation of size {scale}.")
            return

        # Adaptive modes require Hessian
        if type_lower in ["adaptive", "adaptive_stochastic"] and self.most_recent_hessian is None:
            if type_lower == "adaptive":
                if verbose:
                    print("Warning: Adaptive perturbation requested, but no Hessian available. Falling back to 'none'.")
                return  # 'none' means no perturbation
            else:  # adaptive_stochastic
                if verbose:
                    print("Warning: Adaptive stochastic requested, but no Hessian available. Falling back to 'stochastic'.")
                self.perturb(type="stochastic", verbose=verbose)
                return

        # Get the softest mode for adaptive perturbations
        softest_mode, curvature = self.get_softest_hessian_mode(self.position)
        direction = None

        # Mode 3: Deterministic adaptive perturbation
        if type_lower == "adaptive":
            if len(self.trajectory) < 2:
                if verbose:
                    print("Warning: Deterministic 'adaptive' mode needs >1 trajectory point. Defaulting to positive sign.")
                sign = 1.0
            else:
                ref_vector = self.trajectory[-1] - self.trajectory[-2]
                if np.linalg.norm(ref_vector) > 1e-9:
                    dot = np.dot(ref_vector, softest_mode)
                    sign = np.sign(dot) if dot != 0 else 1.0
                else:
                    sign = 1.0
            direction = sign * softest_mode
            if verbose:
                print(f"Applied deterministic adaptive perturbation with alignment sign {sign}.")

        # Mode 4: Randomized adaptive perturbation
        elif type_lower == "adaptive_stochastic":
            sign = 1.0 if np.random.rand() < 0.5 else -1.0
            direction = sign * softest_mode
            if verbose:
                print(f"Applied randomized adaptive perturbation with sign {sign}.")

        else:
            print(f"Warning: Unrecognized perturbation type '{type}'. Falling back to 'stochastic'.")
            self.perturb(type="stochastic", verbose=verbose)
            return

        # Determine perturbation magnitude (scale) for all adaptive modes
        if self.scale_perturb_by_curvature:
            if curvature <= 1e-9:  # Avoid division by zero
                if verbose:
                    print(f"Warning: Non-positive or zero curvature ({curvature:.4f}). Using default size.")
                scale = self.default_perturbation_size
            else:
                if self.use_ema_adaptive_scaling:
                    log_scale = np.log(self.curvature_perturbation_scale / np.sqrt(curvature))
                    if self.log_running_perturb_ema is None:
                        self.log_running_perturb_ema = log_scale
                    else:
                        self.log_running_perturb_ema = (
                            self.ema_alpha * log_scale +
                            (1 - self.ema_alpha) * self.log_running_perturb_ema
                        )
                    proposed_scale = np.exp(self.log_running_perturb_ema)
                else:
                    proposed_scale = self.curvature_perturbation_scale / np.sqrt(curvature)
                scale = np.clip(proposed_scale, self.min_perturbation_size, self.max_perturbation_size)
                if verbose:
                    print(f"Proposed perturbation distance: {proposed_scale:.4f}, clipped to: {scale:.4f}")
        else:
            scale = self.default_perturbation_size

        # Apply the final perturbation
        self.position += scale * direction
    
    def run(self, optimizer=None, max_iterations=100, verbose=True, verbose_opt=False,
            save_summary=False, summary_file=None, stopping_minima_number=None, 
            ignore_max_steps_on_initial_minimization=True):
        """
        Run the ABC simulation.
        
        Args:
            max_iterations: Maximum number of iterations
            verbose: Whether to print progress
            kwargs: Override any default parameters
        """
        
        if optimizer is None:
            optimizer = FIREOptimizer(self)
        self.optimizer = optimizer

        try:         
            for iteration in range(self.current_iteration, max_iterations):
                
                if ignore_max_steps_on_initial_minimization and iteration == 0: 
                    converged = self.descend(verbose=verbose_opt)
                else: 
                    converged = self.descend(verbose=verbose_opt)

                pos = self.position.copy()

                self.perturb(type=self.perturb_type, verbose=verbose)

                self.deposit_bias(pos, verbose=verbose)
                    
                self.update_records()

                if stopping_minima_number is not None and len(self.minima) >= stopping_minima_number: 
                    break  
                    
                print(f"Iteration {iteration+1}/{max_iterations}: "
                    f"Descent converged: {converged}, "
                    # f"Position:{self.position}, "
                    f"Unbiased Energy: {self.unbiased_energies[-1]}",
                    f"Biased Energy: {self.biased_energies[-1]}")
                print()

            print(f"Simulation completed.\n")
        except KeyboardInterrupt:
            print("Simulation Interrupted.\n")
        
        try: 
            self.summarize(save=save_summary, filename=summary_file)
        except Exception as e:
            print(f"Unable to complete summary:\n{e}")

    def summarize(self, save=False, filename=None):
        """Print a comprehensive summary of the optimization results and optionally save to file."""
        output = []

        # 1. Report all found minima with energies
        if self.minima:
            output.append("\nEnergies of found minima:")
            for i, min_pos in enumerate(self.minima):
                idx = np.argmin(np.linalg.norm(np.array(self.trajectory) - np.array(min_pos), axis=1))
                output.append(f"Minimum {i+1}: Position = {min_pos}, Energy = {self.unbiased_energies[idx]}")

        # 3. Approximate saddle points between minima
        if len(self.minima) > 1:
            minima_indices = self.min_indices

            saddle_info = []
            for i in range(len(minima_indices) - 1):
                saddle_info.append((self.saddles[i],
                    self.unbiased_energies[self.saddle_indices[i]]
                ))

            output.append("\nApproximate saddle points between minima:")
            for i, (pos, energy) in enumerate(saddle_info):
                output.append(f"Saddle {i+1}: Position = {pos}, Energy = {energy}")

        # 4. Computational statistics
        output.append("\nComputational statistics:")
        output.append(f"\tTotal energy calls: {self.potential.energy_calls}")
        output.append(f"\tEnergy calls at each min: {self.energy_calls_at_each_min}")
        output.append(f"\tTotal force calls: {self.potential.force_calls}")
        output.append(f"\tForce calls at each min: {self.force_calls_at_each_min}")
        output.append(f"\tTotal steps: {len(self.trajectory)}")
        output.append(f"\tTotal Biases: {len(self.bias_list)}")

        # 5. Reference minima verification
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

        # 6. Reference saddle verification
        if hasattr(self.potential, "known_saddles"):
            true_saddles = self.potential.known_saddles()
            found_saddles = np.array(self.saddles)

            if len(found_saddles) > 0:
                if found_saddles.ndim == 1:
                    found_saddles = found_saddles.reshape(-1, 1)

                matched_saddles = [np.any(np.linalg.norm(found_saddles - true_sad.reshape(1, -1), axis=1) < 0.05) 
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

        # Print summary to terminal
        print('\n'.join(output))

        # Save to file if requested
        if save:
            from datetime import datetime
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"abc_summary_{timestamp}.txt"

            with open(filename, "w") as f:
                # Write summary output
                f.write('\n'.join(output))

                # Write actual config values
                f.write("\n\nConfiguration parameters:\n")
                config_items = {
                    "self.potential": self.potential,
                    "self.curvature_method": self.curvature_method,
                    "self.dump_every": self.dump_every,
                    "self.perturb_type": self.perturb_type,
                    "self.scale_perturb_by_curvature": self.scale_perturb_by_curvature,
                    "self.default_perturbation_size": self.default_perturbation_size,
                    "self.curvature_perturbation_scale": self.curvature_perturbation_scale,
                    "self.bias_height_type": self.bias_height_type,
                    "self.default_bias_height": self.default_bias_height,
                    "self.max_bias_height": self.max_bias_height,
                    "self.curvature_bias_height_scale": self.curvature_bias_height_scale,
                    "self.max_perturbation_size": self.max_perturbation_size,
                    "self.bias_covariance_type": self.bias_covariance_type,
                    "self.default_bias_covariance": self.default_bias_covariance,
                    "self.max_bias_covariance": self.max_bias_covariance,
                    "self.curvature_bias_covariance_scale": self.curvature_bias_covariance_scale,
                    "self.descent_convergence_threshold": self.descent_convergence_threshold,
                    "self.max_descent_steps": self.max_descent_steps,
                    "self.max_acceptable_force_mag": self.max_acceptable_force_mag,
                    "self.optimizer": self.optimizer,
                }

                for key, val in config_items.items():
                    try:
                        f.write(f"{key} = {val}\n")
                    except Exception as e:
                        f.write(f"{key} = <error: {e}>\n")

            print(f"\nSummary saved to '{filename}'")