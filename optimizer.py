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
        
        if self.method.startswith('trust-'):
            result['hess_inv'] = self._hess_approx.get_matrix()
        elif self.method.lower() == "bfgs":
            result['hess_inv'] = self._result.hess_inv 
        elif self.method.lower() == "l-bfgs-b":
            result['hess_inv'] = self.construct_l_bfgs_hess_inv(self._result.hess_inv)
        
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
        self.accepted_positions = []  # Track actual accepted positions
        self._last_accepted_pos = None  # Track last accepted position

    def _run_optimization(self, x0, max_steps=None, convergence_threshold=None):
        from ase.optimize import OPTIMIZER_CLASSES
        from ase import Atoms
        
        self._reset_state()
        self.accepted_positions = []
        self._last_accepted_pos = None
        
        # Setup atoms object
        if isinstance(self.abc_sim.potential, ASEPotentialEnergySurface):
            atoms = self.abc_sim.potential.atoms
            atoms.set_positions(x0.reshape(-1, 3))
        else:
            n_atoms = len(x0) // 3
            print(f"Warning: potential is not of type ASEPotentialEnergySurface. Creating dummy ASE Atoms with {n_atoms} particles")
            if len(x0) % 3 != 0:
                raise ValueError("Position vector must be divisible by 3")
            atoms = Atoms('H' * n_atoms, positions=x0.reshape(-1, 3))
            atoms.calc = _ASECalculatorWrapper(self)

        # Initial evaluation
        self._compute(x0)
        self._register_accepted_step(x0)

        # Configure convergence
        if convergence_threshold is not None:
            self.ase_optimizer_kwargs['fmax'] = convergence_threshold

        # Initialize optimizer
        OptimizerClass = OPTIMIZER_CLASSES[self.optimizer_class]
        self._ase_optimizer = OptimizerClass(atoms, **self.ase_optimizer_kwargs)

        # Callback for tracking accepted steps
        def callback():
            current_pos = atoms.get_positions().flatten()
            energy, _ = self._compute(current_pos)  # Uses caching
            
            # Only register if different from last accepted step
            if (self._last_accepted_pos is None or 
                not np.allclose(current_pos, self._last_accepted_pos, atol=1e-10)):
                self._register_accepted_step(current_pos)

        self._ase_optimizer.attach(callback)

        # Run optimization
        if max_steps is not None:
            self._ase_optimizer.run(steps=max_steps)
        else:
            self._ase_optimizer.run()

        final_pos = atoms.get_positions().flatten()
        return {
            'x': final_pos,
            'nsteps': len(self.accepted_positions),
            'converged': self._ase_optimizer.converged(),
            'used_dummy_atoms': not isinstance(self.abc_sim.potential, ASEPotentialEnergySurface)
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
        x = atoms.get_positions().flatten()
        energy, neg_grad = self.optimizer._compute(x)  # Uses the caching
        
        self.results = {
            'energy': energy,
            'forces': neg_grad.reshape(-1, 3)  # Convert back to ASE format
        }