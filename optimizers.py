from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy

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
        if self._last_x is None or not np.array_equal(x, self._last_x):
            self._last_x = x.copy()

            # Your existing computation logic
            unbiased_energy = self.abc_sim.potential.potential(x)
            biased_energy = self.abc_sim.compute_biased_potential(x, deepcopy(unbiased_energy)) # reuse unbiased call from before
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
        
        return self.biased_energies[-1], -self.biased_forces[-1]  # (energy, gradient)


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
        return {
            'trajectory': [self.trajectory[i] for i in indices],
            'unbiased_energies': [self.unbiased_energies[i] for i in indices],
            'biased_energies': [self.biased_energies[i] for i in indices],
            'unbiased_forces': [self.unbiased_forces[i] for i in indices],
            'biased_forces': [self.biased_forces[i] for i in indices]
        }

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
import numpy as np

class ScipyOptimizer(Optimizer):
    def __init__(self, abc_sim, method='BFGS', **kwargs):
        """
        Unified SciPy optimizer supporting all but trust region methods
        
        Args:
            method: Any SciPy method name
            kwargs: Method-specific options
        """
        super().__init__(abc_sim)
        self.method = method.lower()
        self.optimizer_kwargs = kwargs
        self._result = None
        self.accepted_steps = []

    def _run_optimization(self, x0, max_steps=None, convergence_threshold=None):
        self._reset_state()
        self.accepted_steps = [x0.copy()]
        last_accepted_x = x0.copy()
        
        # Common options
        options = self.optimizer_kwargs.copy()
        if max_steps is not None:
            options['maxiter'] = max_steps

        # Set convergence threshold for different methods
        if convergence_threshold is not None:
            if self.method.upper() in ['L-BFGS-B', 'TNC', 'SLSQP']:
                options['ftol'] = convergence_threshold
            elif self.method.upper() in ['BFGS', 'CG', 'NEWTON-CG']:
                options['gtol'] = convergence_threshold
            elif self.method.upper() == 'POWELL':
                options['ftol'] = convergence_threshold
                options['xtol'] = convergence_threshold

        # Trust-region specific setup
        if self.method.startswith('trust-'):
            def hessp(x, p):
                self._hess_approx.update(x, self._result.grad)
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
        return self._package_result()
    
    from scipy.sparse.linalg import LinearOperator
    def _construct_l_bfgs_hess_inv(self, hess_inv_operator: LinearOperator):
        """Convert L-BFGS-B LinearOperator inverse Hessian to dense matrix"""
        n = self.abc_sim.dimension
        I = np.eye(n)
        
        # Apply the LinearOperator to each basis vector
        inv_H = np.column_stack([hess_inv_operator.matvec(I[:, i]) for i in range(n)])
        
        # Symmetrize the result
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
        
        # Handle Hessian information differently for each method
        if hasattr(self._result, 'hess_inv'):
            if self.method == 'bfgs':
                # BFGS provides dense matrix directly
                result['hess_inv'] = self._result.hess_inv
            elif self.method == 'l-bfgs-b':
                # L-BFGS-B needs conversion from LinearOperator
                result['hess_inv'] = self._construct_l_bfgs_hess_inv(self._result.hess_inv)
        
        return result

    def _accepted_steps(self):
        return self.accepted_steps.copy()

    @property 
    def inv_hessian(self):
        """Get inverse Hessian approximation if available"""
        if not hasattr(self, '_result'):
            raise AttributeError("Optimization not yet run")
        
        if self.method == 'bfgs' and hasattr(self._result, 'hess_inv'):
            return self._result.hess_inv
        elif self.method == 'l-bfgs-b' and hasattr(self._result, 'hess_inv'):
            return self._construct_l_bfgs_hess_inv(self._result.hess_inv)
        
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
        self.convergence_threshold = None
        self._current_atoms = None  # Track current ASE atoms object

    def _run_optimization(self, x0, max_steps=None, convergence_threshold=None):
        from ase import Atoms
        from ase.optimize import BFGS, LBFGS, FIRE, GPMin, BFGSLineSearch, MDMin
        
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
            atoms = deepcopy(self.abc_sim.potential.atoms)
            atoms.set_positions(padded_x0.reshape(-1, 3))
            atoms.calc = _ASECalculatorWrapper(self)
        else:
            n_atoms = len(padded_x0) // 3
            print(f"Warning: potential is not of type ASEPotentialEnergySurface. Creating dummy ASE Atoms with {n_atoms} particles")
            atoms = Atoms('H' * n_atoms, positions=padded_x0.reshape(-1, 3))
            atoms.calc = _ASECalculatorWrapper(self)

        self._current_atoms = atoms  # Store for callback access
        
        # Initial evaluation (with original x0)
        initial_energy, initial_forces = self._compute(x0)
        self._register_accepted_step(x0)

        # Select optimizer class
        optimizer_mapping = {
            'BFGS': BFGS,
            'LBFGS': LBFGS,
            'GPMin': GPMin,
            'FIRE': FIRE,
            'MDMin': MDMin,
            'BFGSLineSearch': BFGSLineSearch,
        }
        OptimizerClass = optimizer_mapping.get(self.optimizer_class, BFGS)

        # Initialize optimizer
        optimizer_kwargs = self.ase_optimizer_kwargs.copy()
        self._ase_optimizer = OptimizerClass(atoms, **optimizer_kwargs)

        # Callback for tracking accepted steps - NO COMPUTE CALLS HERE
        def callback():
            # Get position directly from ASE's internal state
            current_padded_pos = self._current_atoms.get_positions().flatten()
            
            if self._dummy_padding > 0:
                current_pos = current_padded_pos[:-self._dummy_padding]
            else:
                current_pos = current_padded_pos
            
            # Check if we've moved significantly
            if (self._last_accepted_pos is None or 
                not np.allclose(current_pos, self._last_accepted_pos, atol=1e-10)):
                self._register_accepted_step(current_pos)

        self._ase_optimizer.attach(callback)

        # Run optimization
        converged = False
        try:
            if max_steps is not None:
                converged = self._ase_optimizer.run(
                    fmax=self.convergence_threshold, 
                    steps=max_steps
                )
            else:
                converged = self._ase_optimizer.run(
                    fmax=self.convergence_threshold
                )
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            converged = False

        final_padded_pos = atoms.get_positions().flatten()
        if self._dummy_padding > 0:
            final_pos = final_padded_pos[:-self._dummy_padding]
        else:
            final_pos = final_padded_pos
            
        # Ensure final position is computed and recorded
        self._compute(final_pos)
        self._register_accepted_step(final_pos)
        
        # Verify convergence
        # if converged and self.convergence_threshold is not None:
        #     _, forces = self._compute(final_pos)
        #     max_force = np.max(np.abs(forces))
        #     if max_force > self.convergence_threshold * 1.1:
        #         print(f"Warning: Optimizer reported convergence but max force ({max_force}) "
        #               f"exceeds threshold ({self.convergence_threshold})")
        #         converged = False

        return {
            'x': final_pos,
            'nsteps': len(self.accepted_positions),
            'converged': converged,
            'used_dummy_atoms': not isinstance(self.abc_sim.potential, ASEPotentialEnergySurface),
            'used_dummy_dimensions': self._dummy_padding
        }
    
    def _register_accepted_step(self, pos):
        """Register step without duplicating trajectory data"""
        self.accepted_positions.append(pos.copy())
        self._last_accepted_pos = pos.copy()
        
        # Ensure this position is in our trajectory
        # This will compute if needed, or use cached if already computed
        self._compute(pos)

    def _accepted_steps(self):
        return self.accepted_positions.copy()

from ase.calculators.calculator import Calculator, all_changes

class _ASECalculatorWrapper(Calculator):
    """ASE Calculator wrapper to interface your optimizer compute method."""

    implemented_properties = ['energy', 'forces']

    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.results = {}

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        if atoms is None:
            atoms = self.optimizer._current_atoms
        padded_x = atoms.positions.flatten()
        if self.optimizer._dummy_padding > 0:
            x = padded_x[:-self.optimizer._dummy_padding]
        else:
            x = padded_x

        energy, neg_grad = self.optimizer._compute(x)

        if self.optimizer._dummy_padding > 0:
            padded_grad = np.concatenate([neg_grad, np.zeros(self.optimizer._dummy_padding)])
        else:
            padded_grad = neg_grad

        self.results = {
            'energy': energy,
            'forces': -padded_grad.reshape(-1, 3)  # ASE forces = -gradient
        }
        super().calculate(atoms, properties, system_changes)

    def get_potential_energy(self, atoms=None, **kwargs):
        if 'energy' not in self.results:
            self.calculate(atoms)
        return self.results.get('energy')

    def get_forces(self, atoms=None, **kwargs):
        if 'forces' not in self.results:
            self.calculate(atoms)
        return self.results.get('forces')

class SimpleGradientDescent(Optimizer):
    """The cheapest possible gradient descent optimizer"""
    
    def __init__(self, abc_sim, step_size=0.1):
        super().__init__(abc_sim)
        self.step_size = step_size
        self._accepted_positions = []
    
    def _run_optimization(self, x0, max_steps=None, convergence_threshold=None):
        x = x0.copy()
        self._accepted_positions = [x.copy()]
        
        for step in range(max_steps):
            # Compute energy and forces (gradient is -force)
            energy, gradient = self._compute(x)
            
            # Check convergence
            if np.linalg.norm(gradient) < convergence_threshold:
                break
                
            # Take simple gradient descent step
            x = x + self.step_size * gradient
            self._accepted_positions.append(x.copy())
            
        return {
            'x': x,
            'energy': energy,
            'gradient': gradient,
            'nsteps': step + 1,
            'converged': step < max_steps - 1
        }
    
    def _accepted_steps(self):
        return self._accepted_positions
    
class FastLocalLBFGS(Optimizer):
    def __init__(self, abc_sim, init_step_size=0.5, shrink_factor=0.5, **kwargs):
        """
        Accelerated local optimizer that:
        1. Uses L-BFGS-B for fast convergence within basins
        2. Dynamically shrinks steps that would escape the basin
        3. Auto-resets step size after successful steps
        """
        super().__init__(abc_sim)
        self.init_step_size = init_step_size  # Initial trust region size
        self.shrink_factor = shrink_factor    # How aggressively to clip escaping steps
        self.optimizer_kwargs = kwargs
        self.accepted_steps = []

    def _run_optimization(self, x0, max_steps=None, convergence_threshold=None):
        self._reset_state()
        self.accepted_steps = [x0.copy()]
        x = x0.copy()
        current_step_size = self.init_step_size
        
        options = self.optimizer_kwargs.copy()
        if max_steps is not None:
            options['maxiter'] = max_steps

        for _ in range(max_steps or 1000):
            # Run L-BFGS-B with step control
            result = minimize(
                fun=lambda x: self._compute(x)[0],
                x0=x,
                method='L-BFGS-B',
                jac=lambda x: self._compute(x)[1],
                tol=convergence_threshold,
                options={**options, 'maxls': 20},  # Limit line searches
                callback=self._adaptive_step_callback
            )
            
            proposed_x = result.x
            step = proposed_x - x
            step_norm = np.linalg.norm(step)
            
            # Accept step only if it doesn't exceed current trust region
            if step_norm <= current_step_size:
                x = proposed_x
                current_step_size = self.init_step_size  # Reset trust region
            else:
                x = x + (step / step_norm) * current_step_size
                current_step_size *= self.shrink_factor  # Shrink trust region
            
            self.accepted_steps.append(x.copy())
            
            # Check convergence
            _, grad = self._compute(x)
            if np.linalg.norm(grad) < (convergence_threshold or 1e-6):
                break

        return {
            'x': x,
            'energy': self._compute(x)[0],
            'gradient': grad,
            'nsteps': len(self.accepted_steps),
            'converged': np.linalg.norm(grad) < (convergence_threshold or 1e-6)
        }

    def _adaptive_step_callback(self, xk):
        """Track steps without slowing down L-BFGS-B"""
        if not np.allclose(xk, self.accepted_steps[-1]):
            self.accepted_steps.append(xk.copy())

    def _accepted_steps(self):
        return self.accepted_steps.copy()