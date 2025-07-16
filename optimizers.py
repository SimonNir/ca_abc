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
        self._x_to_index = {}

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
           
            # Record trajectory and map
            self.trajectory.append(x.copy())
            index = len(self.trajectory) - 1
            key = tuple(np.round(x, 8))  # Rounded for precision-safe lookup
            self._x_to_index[key] = index

            self.unbiased_energies.append(unbiased_energy)
            self.biased_energies.append(biased_energy)
            self.unbiased_forces.append(unbiased_force)
            self.biased_forces.append(biased_force)
        
        return self.biased_energies[-1], -self.biased_forces[-1].reshape(-1)  # (energy, gradient)


    def get_traj_data(self):
        accepted = self._accepted_steps()
        indices = []

        for pos in accepted:
            key = tuple(np.round(pos, 8))
            index = self._x_to_index.get(key, None)
            if index is not None:
                indices.append(index)
            else:
                print(f"Warning: Accepted position {pos} not found in trajectory (key: {key})")

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
            if not np.allclose(x, last_accepted_x, 1e-10):
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
            if self.method.lower() == 'bfgs':
                # BFGS provides dense matrix directly
                result['hess_inv'] = self._result.hess_inv
            elif self.method.lower() == 'l-bfgs-b':
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
from ase.optimize import BFGS, LBFGS, FIRE, GPMin, BFGSLineSearch, MDMin

class ASEOptimizer(Optimizer):
    def __init__(self, abc_sim, optimizer_class='BFGS', **ase_optimizer_kwargs):
        super().__init__(abc_sim)
        self.optimizer_class = optimizer_class
        self.ase_optimizer_kwargs = ase_optimizer_kwargs
        self._ase_optimizer = None
        self.accepted_positions = []
        self._last_accepted_pos = None
        self.convergence_threshold = None
        self._current_atoms = None
        self._result = None  # For consistency with ScipyOptimizer

    def _run_optimization(self, x0, max_steps=None, convergence_threshold=None):        
        self._reset_state()
        self.accepted_positions = []
        self._last_accepted_pos = None
        self.convergence_threshold = convergence_threshold

        # Handle both regular and canonical PES cases
        if isinstance(self.abc_sim.potential, ASEPotentialEnergySurface):
            atoms = self.abc_sim.potential.atoms.copy()
            atoms.set_positions(x0.reshape(-1, 3))
        # elif hasattr(self.abc_sim.potential, 'atoms'):  # For canonical PES
            # atoms = self.abc_sim.potential.atoms.copy()
        #     atoms.set_positions(x0.reshape(-1, 3))
        else:
            # Fallback for non-ASE potentials
            from ase import Atoms
            atoms = Atoms('H' * (len(x0)//3), positions=x0.reshape(-1, 3))

        atoms.calc = _ASECalculatorWrapper(self)
        self._current_atoms = atoms

        self._register_accepted_step(x0)

        optimizer_mapping = {
            'BFGS': BFGS,
            'LBFGS': LBFGS,
            'GPMin': GPMin,
            'FIRE': FIRE,
            'MDMin': MDMin,
            'BFGSLineSearch': BFGSLineSearch,
        }
        
        OptimizerClass = optimizer_mapping.get(self.optimizer_class, BFGS)
        optimizer_kwargs = self.ase_optimizer_kwargs.copy()

            
        self._ase_optimizer = OptimizerClass(atoms, **optimizer_kwargs)

        def callback():
            current_pos = self._current_atoms.get_positions().flatten()
            if (self._last_accepted_pos is None or 
                not np.allclose(current_pos, self._last_accepted_pos, atol=1e-10)):
                self._register_accepted_step(current_pos)

        self._ase_optimizer.attach(callback)

        try:
            converged = self._ase_optimizer.run(
                fmax=convergence_threshold if convergence_threshold else 0.05,
                steps=max_steps if max_steps else 1000
            )
            message = "Optimization converged" if converged else "Optimization did not converge"
        except Exception as e:
            print(f"Optimization failed: {e}")
            converged = False
            message = str(e)

        final_pos = self._current_atoms.get_positions().flatten()
        self._compute(final_pos)
        self._register_accepted_step(final_pos)

        # Package result similar to ScipyOptimizer for consistency
        self._result = {
            'x': final_pos,
            'success': converged,
            'message': message,
            'nit': len(self.accepted_positions),
            'hess_inv': None  # ASE optimizers don't provide Hessian info
        }
        
        return self._package_result()

    def _package_result(self):
        """Standardized result format matching ScipyOptimizer"""
        if not hasattr(self, '_result'):
            raise RuntimeError("Optimization not yet run")
            
        return {
            'x': self._result['x'],
            'converged': self._result['success'],
            'nit': self._result['nit'],
            'message': self._result['message']
            # ASE doesn't provide Hessian information
        }

    def _register_accepted_step(self, pos):
        self.accepted_positions.append(pos.copy())
        self._last_accepted_pos = pos.copy()

    def _accepted_steps(self):
        return self.accepted_positions.copy()
    
from ase.calculators.calculator import Calculator, all_changes
import numpy as np


class _ASECalculatorWrapper(Calculator):
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.results = {}

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        if atoms is None:
            atoms = self.optimizer._current_atoms
        
        # Get current position and compute through ABC's cached _compute()
        current_pos = atoms.get_positions().flatten()
        energy, neg_grad = self.optimizer._compute(current_pos)
        
        # Store results in ASE-expected format
        self.results = {
            'energy': energy,
            'forces': -neg_grad.reshape(-1, 3)  # ASE expects forces = -gradient
        }
        
        super().calculate(atoms, properties, system_changes)

    def get_potential_energy(self, atoms=None, **kwargs):
        if atoms is not None:
            self.calculate(atoms)
        return self.results.get('energy', 0.0)

    def get_forces(self, atoms=None, **kwargs):
        if atoms is not None:
            self.calculate(atoms)
        return self.results.get('forces', np.zeros((len(atoms), 3)))

    def get_stress(self, atoms=None, **kwargs):
        # Dummy implementation required by some ASE optimizers
        return np.zeros(6)
    

from potentials import CanonicalASEPES, internal_to_cartesian, cartesian_to_internal
from ase.optimize import BFGS, LBFGS, FIRE, GPMin, BFGSLineSearch, MDMin
    
class CanonicalASEOptimizer(ASEOptimizer):
    def __init__(self, abc_sim, optimizer_class='BFGS', **ase_optimizer_kwargs):
        super().__init__(abc_sim, optimizer_class, **ase_optimizer_kwargs)
        
    def _setup_constraints(self, atoms):
        """Apply constraints to maintain canonical frame during optimization"""
        from ase.constraints import FixedCartesian, FixCartesian
        
        constraints = [
            FixCartesian(0, [True, True, True]),   # Fix atom 0 completely
            FixCartesian(1, [False, True, True]),  # Atom 1: only x movable
        ]
        
        if len(atoms) > 2:
            constraints.append(FixCartesian(2, [False, False, True]))  # Atom 2: x,y movable
        
        atoms.set_constraint(constraints)
        return atoms
    

    def _run_optimization(self, x0, max_steps=None, convergence_threshold=None):
        # Convert internal to Cartesian
        cartesian_pos = internal_to_cartesian(x0, self.abc_sim.potential.N, 
                                            self.abc_sim.potential.k_soft)
        
        # Create and constrain atoms
        atoms = self.abc_sim.potential.atoms.copy()
        atoms.set_positions(cartesian_pos.reshape(-1, 3))
        atoms = self._setup_constraints(atoms)
        
        # Use our specialized wrapper
        atoms.calc = _CanonicalASECalculatorWrapper(self)
        self._current_atoms = atoms

        # Initialize optimizer
        optimizer_mapping = {
            'BFGS': BFGS,
            'LBFGS': LBFGS,
            'GPMin': GPMin,
            'FIRE': FIRE,
            'MDMin': MDMin,
            'BFGSLineSearch': BFGSLineSearch,
        }
        OptimizerClass = optimizer_mapping.get(self.optimizer_class, BFGS)
        
        optimizer_kwargs = self.ase_optimizer_kwargs.copy()
        if max_steps is not None:
            optimizer_kwargs['steps'] = max_steps
            
        self._ase_optimizer = OptimizerClass(atoms, **optimizer_kwargs)

        # Track accepted steps
        self.accepted_positions = []
        self._last_accepted_pos = None
        self._register_accepted_step(x0)

        # Callback to track accepted steps
        def callback():
            current_cart = self._current_atoms.get_positions().flatten()
            current_internal = cartesian_to_internal(current_cart, self.abc_sim.potential.k_soft)
            if (self._last_accepted_pos is None or 
                not np.allclose(current_internal, self._last_accepted_pos, atol=1e-10)):
                self._register_accepted_step(current_internal)

        self._ase_optimizer.attach(callback)

        # Run optimization
        try:
            converged = self._ase_optimizer.run(
                fmax=convergence_threshold if convergence_threshold else 0.05,
                steps=max_steps if max_steps else 1000
            )
            message = "Optimization converged" if converged else "Optimization did not converge"
        except Exception as e:
            print(f"Optimization failed: {e}")
            converged = False
            message = str(e)

        # Get final results
        final_cart = self._current_atoms.get_positions().flatten()
        final_internal = cartesian_to_internal(final_cart, self.abc_sim.potential.k_soft)
        self._compute(final_internal)
        self._register_accepted_step(final_internal)

        # Package results
        self._result = {
            'x': final_internal,
            'success': converged,
            'message': message,
            'nit': len(self.accepted_positions),
            'hess_inv': None  # ASE doesn't provide Hessian info
        }
        
        return self._package_result()
    

from potentials import softplus, d_softplus_dx

class _CanonicalASECalculatorWrapper(Calculator):
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.results = {}
        self.k_soft = optimizer.abc_sim.potential.k_soft
        self.N = optimizer.abc_sim.potential.N

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        if atoms is None:
            atoms = self.optimizer._current_atoms
        
        # Get current Cartesian positions
        current_cart = atoms.get_positions().flatten()
        
        # Convert to internal coordinates for energy/force computation
        current_internal = cartesian_to_internal(current_cart, self.k_soft)
        
        # Get energy and forces in internal coordinates (3N-6 dimensional)
        energy, neg_grad_internal = self.optimizer._compute(current_internal)
        
        # Convert internal forces back to Cartesian space
        # 1. Create full 3N gradient with zeros for constrained dimensions
        full_grad = np.zeros(3*self.N)
        
        # For N >= 2:
        if self.N >= 2:
            # Atom 1 x-component (constrained to be positive via softplus)
            full_grad[3] = neg_grad_internal[0] * d_softplus_dx(current_internal[0], self.k_soft)
            
        # For N >= 3:
        if self.N >= 3:
            # Atom 2 x,y components
            full_grad[6] = neg_grad_internal[1]  # x (unconstrained)
            full_grad[7] = neg_grad_internal[2] * d_softplus_dx(current_internal[2], self.k_soft)  # y (softplus)
            
        # For N >= 4:
        if self.N >= 4:
            # Remaining atoms (x,y unconstrained, z softplus)
            internal_idx = 3
            for atom_idx in range(3, self.N):
                # x component
                full_grad[3*atom_idx] = neg_grad_internal[internal_idx]
                # y component
                full_grad[3*atom_idx+1] = neg_grad_internal[internal_idx+1]
                # z component (softplus)
                full_grad[3*atom_idx+2] = neg_grad_internal[internal_idx+2] * d_softplus_dx(current_internal[internal_idx+2], self.k_soft)
                internal_idx += 3
        
        # Store results in ASE format (Cartesian forces)
        self.results = {
            'energy': energy,
            'forces': -full_grad.reshape(-1, 3)  # ASE expects forces = -gradient
        }
        
        super().calculate(atoms, properties, system_changes)

    def get_potential_energy(self, atoms=None, **kwargs):
        if atoms is not None:
            self.calculate(atoms)
        return self.results.get('energy', 0.0)

    def get_forces(self, atoms=None, **kwargs):
        if atoms is not None:
            self.calculate(atoms)
        return self.results.get('forces', np.zeros((self.N, 3)))

    def get_stress(self, atoms=None, **kwargs):
        return np.zeros(6)
    

def manual_bfgs_inverse_hessian(positions, forces):
    """
    Approximates the inverse Hessian at the final position using BFGS updates.

    Parameters:
    - positions: list of np.ndarray of shape (n,)
    - forces: list of np.ndarray of shape (n,) (assumed to be negative gradients)

    Returns:
    - H: np.ndarray of shape (n, n), the approximate inverse Hessian
    """
    if len(positions) != len(forces):
        raise ValueError("positions and forces must be the same length")
    if len(positions) < 2:
        raise ValueError("Need at least two position-force pairs")

    n = positions[0].shape[0]
    H = np.eye(n)  # initial inverse Hessian

    for i in range(len(positions) - 1):
        s = positions[i+1] - positions[i]       # step
        y = -(forces[i+1] - forces[i])             # gradient difference (grad = -force)
        
        ys = np.dot(y, s)
        if ys <= 1e-50:
            # print(ys)
            # Skip update if curvature condition fails
            # print("curvature condition not satisfied; hess_inv update skipped")
            if ys > 0: 
                print(f"Hessian Update Warning: ys of {ys} skipped for being too small in magnitude")
            continue
        
        rho = 1.0 / ys
        I = np.eye(n)
        Hy = H @ y
        outer_ss = np.outer(s, s)
        outer_sy = np.outer(s, y)
        outer_ys = np.outer(y, s)
        outer_Hy_yH = np.outer(Hy, Hy)

        H = (I - rho * outer_sy) @ H @ (I - rho * outer_ys) + rho * outer_ss

    return H

import numpy as np
from scipy.optimize._hessian_update_strategy import BFGS
from copy import deepcopy

def bfgs_inverse_hessian(positions, forces, assume_forces_are_neg_grads=True):
    if len(positions) != len(forces):
        raise ValueError("positions and forces must have the same length")
    if len(positions) < 2:
        raise ValueError("Need at least two position-force pairs")

    n = positions[0].shape[0]
    updater = BFGS(exception_strategy='skip_update', init_scale='auto')
    updater.initialize(n, approx_type='inv_hess')  # inverse Hessian approx

    if assume_forces_are_neg_grads:
        grads = [-f for f in forces]
    else:
        grads = forces

    for i in range(len(positions) - 1):
        s = positions[i+1] - positions[i]
        y = grads[i+1] - grads[i]
        updater.update(s, y)

    return updater.get_matrix()

import numpy as np
from scipy.optimize._hessian_update_strategy import BFGS
from copy import deepcopy

def bfgs_inverse_hessian(positions, forces, assume_forces_are_neg_grads=True):
    if len(positions) != len(forces):
        raise ValueError("positions and forces must have the same length")
    if len(positions) < 2:
        raise ValueError("Need at least two position-force pairs")

    n = positions[0].shape[0]
    updater = BFGS(exception_strategy='skip_update', init_scale='auto')
    updater.initialize(n, approx_type='inv_hess')  # inverse Hessian approx

    if assume_forces_are_neg_grads:
        grads = [-f for f in forces]
    else:
        grads = forces

    for i in range(len(positions) - 1):
        s = positions[i+1] - positions[i]
        y = grads[i+1] - grads[i]
        updater.update(s, y)

    return updater.get_matrix()

class FIREOptimizer(Optimizer):
    def __init__(self, abc_sim, dt=0.01, alpha=0.1, dt_max=0.05, N_min=5,
                 f_inc=1.05, f_dec=0.5, alpha_dec=0.95, max_steps=1000,
                 f_tol=1e-4, max_step_size=0.05, velocity_damping=0.9):
        """
        - Smaller dt and dt_max for cautious steps
        - Added max_step_size to clip max displacement per step
        - velocity_damping applied every step to reduce momentum buildup
        """
        super().__init__(abc_sim)
        self.dt = dt
        self.alpha = alpha
        self.dt_max = dt_max
        self.N_min = N_min
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.alpha_dec = alpha_dec
        self.max_steps = max_steps
        self.f_tol = f_tol
        self.max_step_size = max_step_size
        self.velocity_damping = velocity_damping

        self.v = None
        self.accepted_positions = []

    def _run_optimization(self, x0, max_steps=None, convergence_threshold=None):
        x = x0.copy()
        self.v = np.zeros_like(x)
        dt = self.dt
        alpha = self.alpha
        N_min = self.N_min
        f_inc = self.f_inc
        f_dec = self.f_dec
        alpha_dec = self.alpha_dec
        dt_max = self.dt_max
        max_steps = max_steps or self.max_steps
        f_tol = convergence_threshold or self.f_tol
        max_step_size = self.max_step_size
        velocity_damping = self.velocity_damping

        n_pos = 0
        # Don't add initial position here - let the first _compute() call handle it
        self.accepted_positions = [] 

        for step in range(max_steps):
            energy, grad = self._compute(x)
            # Now x is in trajectory, so add it to accepted_positions
            self.accepted_positions.append(x.copy())
            
            force = -grad
            fmax = np.max(np.abs(force))

            if fmax < f_tol:
                break

            self.v += dt * force

            P = np.dot(force, self.v)
            v_norm = np.linalg.norm(self.v)
            f_norm = np.linalg.norm(force)
            if f_norm > 1e-20 and v_norm > 1e-20:
                self.v = (1 - alpha) * self.v + alpha * force * (v_norm / f_norm)

            if P > 0:
                n_pos += 1
                if n_pos > N_min:
                    dt = min(dt * f_inc, dt_max)
                    alpha *= alpha_dec
            else:
                self.v[:] = 0
                dt *= f_dec
                alpha = self.alpha
                n_pos = 0

            # Apply velocity damping to reduce momentum buildup
            self.v *= velocity_damping

            # Proposed step
            step_vec = dt * self.v

            # Clip step size to max_step_size
            step_norm = np.linalg.norm(step_vec)
            if step_norm > max_step_size:
                step_vec = step_vec / step_norm * max_step_size

            x = x + step_vec

        converged = (fmax < f_tol)

        # Compute BFGS inverse Hessian using full trajectory (like SciPy does)
        hess_inv = None
        if len(self.trajectory) >= 2:
            try:
                hess_inv = bfgs_inverse_hessian(self.trajectory, self.biased_forces)
            except Exception as e:
                print(f"Warning: Could not compute BFGS inverse Hessian: {e}")
                hess_inv = None

        return {
            'x': x,
            'energy': energy,
            'converged': converged,
            'nsteps': step + 1,
            'hess_inv': hess_inv
        }

    def _accepted_steps(self):
        return self.accepted_positions.copy()