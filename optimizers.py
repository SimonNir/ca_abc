from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    """Abstract base class for all optimizer backends"""
    
    def __init__(self, abc_sim):
        self.abc_sim = abc_sim
        self._reset_state()

    def _reset_state(self):
        """Initialize/clear all trajectory tracking"""
        
        self.trajectory = []
        self.unbiased_energies = []
        self.biased_energies = []
        self.unbiased_forces = []
        self.biased_forces = []
        self._last_x = None

    def _compute(self, x):
        """Compute and record energy/forces (cached)"""
        if self._last_x is None or not np.allclose(x, self._last_x):
            self._last_x = x.copy()
            
            # Your existing computation logic
            unbiased_energy = self.abc_sim.potential.potential(x)
            biased_energy = self.abc_sim.compute_biased_potential(x, unbiased_energy) # reuse unbiased call from before
            
            try:
                unbiased_force = - self.abc_sim.potential.gradient(x)
            except NotImplementedError:
                unbiased_force = None
            
            biased_force = self.abc_sim.compute_biased_force(x, unbiased_force)
            
            # Record
            self.trajectory.append(x.copy())
            self.unbiased_energies.append(unbiased_energy)
            self.biased_energies.append(biased_energy)
            self.unbiased_forces.append(unbiased_force)
            self.biased_forces.append(biased_force)
        
        return self.biased_energies[-1], -self.biased_forces[-1]  # (energy, -gradient)


    def get_traj_data(self):
        accepted = self._accepted_steps()
        indices = []
        for pos in accepted:
            found = False
            for i, traj_pos in enumerate(self.trajectory):
                if np.allclose(pos, traj_pos, atol=1e-10):
                    indices.append(i)
                    found = True
                    break
            if not found:
                # Try with a higher tolerance
                for i, traj_pos in enumerate(self.trajectory):
                    if np.allclose(pos, traj_pos, atol=1e-8):
                        indices.append(i)
                        found = True
                        print(f"Warning: Position {pos} not found with tol={1e-10}, matched with tol={1e-8} at index {i}.")
                        break
            if not found:
                print(f"Warning: Accepted position {pos} not found in trajectory with any tolerance.")
        traj = [self.trajectory[i] for i in indices]
        unbiased_e = [self.unbiased_energies[i] for i in indices]
        biased_e = [self.biased_energies[i] for i in indices]
        unbiased_f = [self.unbiased_forces[i] for i in indices]
        biased_f = [self.biased_forces[i] for i in indices]
        return (traj, unbiased_e, biased_e, unbiased_f, biased_f)

    def descend(self, x0, max_steps=None, convergence_threshold=None):
        """Universal descent method (same for all backends)"""
        self._reset_state()
        result = self._run_optimization(x0, max_steps, convergence_threshold)
        return result, self.get_traj_data()
    
    @abstractmethod
    def _run_optimization(self, x0, max_steps, convergence_threshold):
        """
        Backend-specific optimization (implemented by child classes)
        Should call _compute()
        """
        pass

    @abstractmethod
    def _accepted_steps():
        """
        Return a list of the positions that were actually accepted by the optimizer, in the order they were accepted
        
        Would be nicer if could get the indices, but many optimizer types (e.g. SciPy) do not allow for this
        """
        pass 
    

from scipy.optimize import minimize
from scipy.optimize._hessian_update_strategy import BFGS

class ScipyOptimizer(Optimizer):
    def __init__(self, abc_sim, method='L-BFGS-B', **kwargs):
        """
        Unified SciPy optimizer supporting:
        - Quasi-Newton methods (L-BFGS-B, BFGS, etc.)
        - Trust-region methods (trust-krylov, trust-ncg)
        
        Args:
            method: Any SciPy method name
            kwargs: Method-specific options
        """
        super().__init__(abc_sim)
        self.method = method
        self.optimizer_kwargs = kwargs
        self._result = None
        self._hess_approx = BFGS() if method.startswith('trust-') else None
        self.accepted_steps = []

    def _run_optimization(self, x0, max_steps=None, convergence_threshold=None):
        self._reset_state()
        self.accepted_steps = [x0.copy()]
        last_accepted_x = x0.copy()
        
        # Common options
        options = self.optimizer_kwargs.copy()
        if max_steps is not None:
            options['maxiter'] = max_steps
        if convergence_threshold is not None:
            if self.method.startswith('trust-'):
                options['gtol'] = convergence_threshold
            else:
                # For L-BFGS-B and similar methods
                options['ftol'] = convergence_threshold

        # Trust-region specific setup
        if self.method.startswith('trust-'):
            def hessp(x, p):
                # Compute function value and gradient at x
                _, grad = self._compute(x)
                self._hess_approx.update(x, grad)
                return self._hess_approx.dot(p)
            extra_args = {'hessp': hessp}
        else:
            extra_args = {}

        def callback(x):
            nonlocal last_accepted_x
            if not np.allclose(x, last_accepted_x):
                self._compute(x)  # Ensure cached
                self.accepted_steps.append(x.copy())
                last_accepted_x = x.copy()

        self._result = minimize(
            fun=lambda x: self._compute(x)[0],
            x0=x0,
            method=self.method,
            jac=lambda x: self._compute(x)[1],
            callback=callback,
            options=options,
            **extra_args
        )

        # Finalize Hessian approximation for trust-region
        if self.method.startswith('trust-') and hasattr(self._result, 'grad'):
            self._hess_approx.update(self._result.x, self._result.grad)

        return self._package_result()
    
    from scipy.sparse.linalg import LinearOperator
    def construct_l_bfgs_hess_inv(self, hess_inv: LinearOperator): 
        n = self.abc_sim.dimension
        I = np.eye(n)

        inv_H = np.column_stack([hess_inv @ I[:, i] for i in range(n)])

        # Symmetrize
        inv_H = 0.5 * (inv_H + inv_H.T)

        return inv_H

    def _package_result(self):
        """Standardized result format"""
        result = {
            'x': self._result.x,
            'converged': self._result.success,
            'nit': self._result.nit,
            'message': self._result.message
        }
        
        try:
            if self.method.startswith('trust-'):
                if self._hess_approx is not None:
                    # Ensure we have a proper matrix
                    hess_inv = self._hess_approx.get_matrix()
                    # Make sure it's 2D
                    if hess_inv.ndim == 1:
                        hess_inv = np.diag(hess_inv)
                    result['hess_inv'] = hess_inv
            elif self.method.lower() == "bfgs":
                if hasattr(self._result, 'hess_inv'):
                    result['hess_inv'] = self._result.hess_inv 
            elif self.method.lower() == "l-bfgs-b":
                if hasattr(self._result, 'hess_inv'):
                    result['hess_inv'] = self.construct_l_bfgs_hess_inv(self._result.hess_inv)
        except Exception as e:
            print(f"Warning: Could not extract Hessian information: {str(e)}")
        
        return result

    def _accepted_steps(self):
        return self.accepted_steps.copy()

    @property 
    def inv_hessian(self):
        """Get inverse Hessian (only available for trust-region methods)"""
        if self.method.startswith('trust-'):
            return self._hess_approx.get_matrix()
        raise AttributeError(f"Inverse Hessian not available for method '{self.method}'")


from potentials import ASEPotentialEnergySurface

class ASEOptimizer(Optimizer):
    def __init__(self, abc_sim, optimizer_class='BFGS', **ase_optimizer_kwargs):
        super().__init__(abc_sim)
        self.optimizer_class = optimizer_class
        self.ase_optimizer_kwargs = ase_optimizer_kwargs
        self._ase_optimizer = None
        self.accepted_positions = []
        self._last_accepted_pos = None
        self._dummy_padding = 0
        self.convergence_threshold = None  # Store separately from optimizer kwargs

    def _run_optimization(self, x0, max_steps=None, convergence_threshold=None):
        from ase import Atoms
        from ase.optimize import BFGS, LBFGS, FIRE, GPMin, BFGSLineSearch, MDMin # Import optimizers directly
        
        self._reset_state()
        self.accepted_positions = []
        self._last_accepted_pos = None
        self.convergence_threshold = convergence_threshold
        
        # Handle non-multiple-of-3 dimensions
        remainder = len(x0) % 3
        if remainder != 0:
            self._dummy_padding = 3 - remainder
            padded_x0 = np.concatenate([x0, np.zeros(self._dummy_padding)])
            print(f"Warning: position vector length {len(x0)} is not divisible by 3. "
                  f"Adding {self._dummy_padding} dummy dimensions (constant zeros).")
        else:
            self._dummy_padding = 0
            padded_x0 = x0

        # Setup atoms object
        if isinstance(self.abc_sim.potential, ASEPotentialEnergySurface):
            atoms = self.abc_sim.potential.atoms
            atoms.set_positions(padded_x0.reshape(-1, 3))
        else:
            n_atoms = len(padded_x0) // 3
            print(f"Warning: potential is not of type ASEPotentialEnergySurface. Creating dummy ASE Atoms with {n_atoms} particles")
            atoms = Atoms('H' * n_atoms, positions=padded_x0.reshape(-1, 3))
            atoms.calc = _ASECalculatorWrapper(self)

        # Initial evaluation (with original x0)
        self._compute(x0)
        self._register_accepted_step(x0)

        # Select optimizer class
        optimizer_mapping = {
            'BFGS': BFGS,
            'LBFGS': LBFGS,
            'GPMin': GPMin,
            'FIRE': FIRE,
            'MDMin': MDMin,
            'BFGSLineSearch': BFGSLineSearch,
            # Add other optimizers as needed
        }
        OptimizerClass = optimizer_mapping.get(self.optimizer_class, BFGS)  # Default to BFGS

        # Remove fmax from kwargs if present (we'll handle it separately)
        optimizer_kwargs = self.ase_optimizer_kwargs.copy()
        optimizer_kwargs.pop('fmax', None)

        # Initialize optimizer
        self._ase_optimizer = OptimizerClass(atoms, **optimizer_kwargs)
        
        # Set convergence threshold if specified
        if self.convergence_threshold is not None:
            self._ase_optimizer.fmax = self.convergence_threshold

        # Callback for tracking accepted steps
        def callback():
            current_padded_pos = atoms.get_positions().flatten()
            if self._dummy_padding > 0:
                current_pos = current_padded_pos[:-self._dummy_padding]
            else:
                current_pos = current_padded_pos
                
            energy, _ = self._compute(current_pos)
            
            if (self._last_accepted_pos is None or 
                not np.allclose(current_pos, self._last_accepted_pos, atol=1e-10)):
                self._register_accepted_step(current_pos)

        self._ase_optimizer.attach(callback)

        # Run optimization
        if max_steps is not None:
            self._ase_optimizer.run(steps=max_steps)
        else:
            self._ase_optimizer.run()

        final_padded_pos = atoms.get_positions().flatten()
        if self._dummy_padding > 0:
            final_pos = final_padded_pos[:-self._dummy_padding]
        else:
            final_pos = final_padded_pos
            
        # Only return converged=True if ASE optimizer converged by force (not just by reaching max steps)
        converged = False
        if hasattr(self._ase_optimizer, 'converged'):
            converged = self._ase_optimizer.converged()
        elif hasattr(self._ase_optimizer, 'converged_by_forces'):
            converged = self._ase_optimizer.converged_by_forces

        # If max_steps was set and we hit the limit, do not report as converged
        if max_steps is not None and self._ase_optimizer.nsteps >= max_steps:
            converged = False

        return {
            'x': final_pos,
            'nsteps': len(self.accepted_positions),
            'converged': converged,
            'used_dummy_atoms': not isinstance(self.abc_sim.potential, ASEPotentialEnergySurface),
            'used_dummy_dimensions': self._dummy_padding > 0
        }

    def _register_accepted_step(self, pos):
        """Helper to register an accepted step"""
        self.accepted_positions.append(pos.copy())
        self._last_accepted_pos = pos.copy()

    def _accepted_steps(self):
        """Return the actual accepted positions"""
        return self.accepted_positions.copy()


class _ASECalculatorWrapper:
    """Private wrapper that ensures all calculations go through _compute()"""
    def __init__(self, optimizer):
        self.optimizer = optimizer
        
    def calculate(self, atoms, properties, system_changes):
        padded_x = atoms.get_positions().flatten()
        if self.optimizer._dummy_padding > 0:
            x = padded_x[:-self.optimizer._dummy_padding]
        else:
            x = padded_x
            
        energy, neg_grad = self.optimizer._compute(x)  # Uses the caching
        
        # Pad the gradient with zeros for dummy dimensions
        if self.optimizer._dummy_padding > 0:
            padded_grad = np.concatenate([neg_grad, np.zeros(self.optimizer._dummy_padding)])
        else:
            padded_grad = neg_grad
            
        self.results = {
            'energy': energy,
            'forces': -padded_grad.reshape(-1, 3)  # Convert back to ASE format
        }


class ConservativeSteepestDescent(Optimizer):
    """Steepest descent optimizer that guarantees no overshooting of local minima"""
    
    def __init__(self, abc_sim, initial_step_size=1.0, min_step_size=1e-10, 
                 max_step_size=1.0, armijo_c1=1e-4, max_line_search_steps=20,
                 friction_coeff=0.1, max_iter=1000, tol=1e-6):
        super().__init__(abc_sim)
        self.initial_step_size = initial_step_size
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.armijo_c1 = armijo_c1
        self.max_line_search_steps = max_line_search_steps
        self.friction_coeff = friction_coeff
        self.max_iter = max_iter
        self.tol = tol
        self._accepted_positions = []
        self._result = None
        
    def _run_optimization(self, x0, max_steps=None, convergence_threshold=None):
        max_steps = max_steps if max_steps is not None else self.max_iter
        convergence_threshold = convergence_threshold if convergence_threshold is not None else self.tol
            
        x = x0.copy()
        current_energy, negative_grad = self._compute(x)
        self._accepted_positions = [x.copy()]
        step_size = self.initial_step_size
        converged = False
        message = 'Maximum number of iterations reached'
        
        for step in range(max_steps):
            grad = -negative_grad
            
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            if grad_norm < convergence_threshold:
                converged = True
                message = f'Optimization converged (gradient norm {grad_norm:.2e} < {convergence_threshold:.2e})'
                break
                
            direction = grad / (grad_norm if grad_norm > 0 else 1)
            
            # Perform conservative line search
            new_x, new_energy, step_size = self._conservative_line_search(
                x, current_energy, grad, direction, step_size
            )
            
            # Apply friction to prevent oscillations
            if len(self._accepted_positions) > 1:
                prev_vec = self._accepted_positions[-1] - self._accepted_positions[-2]
                curr_vec = new_x - self._accepted_positions[-1]
                if np.dot(prev_vec, curr_vec) < 0:
                    step_size *= (1 - self.friction_coeff)
            
            # Update for next iteration
            x = new_x.copy()
            current_energy = new_energy
            _, negative_grad = self._compute(x)
            self._accepted_positions.append(x.copy())
            
            if step_size < self.min_step_size:
                message = 'Step size smaller than minimum allowed'
                break
                
        # Store result as dictionary
        self._result = {
            'x': x,
            'converged': converged,
            'nit': step + 1,
            'message': message,
            'fun': current_energy
        }
        
        return self._package_result()
    
    def _package_result(self):
        """Return the result dictionary directly"""
        if self._result is None:
            raise RuntimeError("Optimization hasn't been run yet")
        return self._result
    
    def _conservative_line_search(self, x, current_energy, grad, direction, initial_step_size):
        """Line search that guarantees energy decrease"""
        step_size = min(initial_step_size, self.max_step_size)
        best_step_size = 0
        best_energy = current_energy
        best_x = x.copy()
        
        for _ in range(self.max_line_search_steps):
            candidate_x = x + step_size * direction
            candidate_energy, _ = self._compute(candidate_x)
            
            required_decrease = self.armijo_c1 * step_size * np.dot(grad, direction)
            
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_step_size = step_size
                best_x = candidate_x.copy()
                
                if (current_energy - candidate_energy) < required_decrease:
                    step_size *= 0.5
                    continue
            else:
                step_size *= 0.5
                continue
                
            break
            
        if best_step_size == 0:
            best_step_size = self.min_step_size
            best_x = x + best_step_size * direction
            best_energy, _ = self._compute(best_x)
            
        return best_x, best_energy, best_step_size
    
    def _accepted_steps(self):
        return self._accepted_positions