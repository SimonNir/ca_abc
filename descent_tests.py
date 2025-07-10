from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize

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

    def get_traj_data(self):
        """Return trajectory data for accepted steps"""
        return {
            'trajectory': self.trajectory.copy(),
            'unbiased_energies': self.unbiased_energies.copy(),
            'biased_energies': self.biased_energies.copy(),
            'unbiased_forces': self.unbiased_forces.copy(),
            'biased_forces': self.biased_forces.copy()
        }

    def descend(self, x0, max_steps=None, convergence_threshold=None):
        """Universal descent method"""
        self._reset_state()
        result = self._run_optimization(x0, max_steps, convergence_threshold)
        return result, self.get_traj_data()
    
    @abstractmethod
    def _run_optimization(self, x0, max_steps, convergence_threshold):
        """Backend-specific optimization implementation"""
        pass


class ScipyOptimizer(Optimizer):
    """Minimal SciPy optimizer that doesn't interfere with optimization"""
    
    def __init__(self, abc_sim, method='BFGS', **kwargs):
        super().__init__(abc_sim)
        self.method = method.upper()
        self.optimizer_kwargs = kwargs
        self._result = None

    def _run_optimization(self, x0, max_steps=None, convergence_threshold=None):
        """Run SciPy optimization with minimal interference"""
        self._reset_state()
        
        # Store trajectory points for post-processing
        self.trajectory_points = []
        
        def objective(x):
            """Pure objective function - no side effects"""
            unbiased_energy = self.abc_sim.potential.potential(x)
            biased_energy = self.abc_sim.compute_biased_potential(x, deepcopy(unbiased_energy))
            return biased_energy

        def gradient(x):
            """Pure gradient function - no side effects"""
            try:
                unbiased_force = -self.abc_sim.potential.gradient(x)
            except NotImplementedError:
                unbiased_force = None
            biased_force = self.abc_sim.compute_biased_force(x, unbiased_force)
            return -biased_force  # SciPy expects gradient of objective

        def callback(x):
            """Store trajectory points for later processing"""
            self.trajectory_points.append(x.copy())

        # Setup options
        options = self.optimizer_kwargs.copy()
        if max_steps is not None:
            options['maxiter'] = max_steps

        # Set convergence thresholds
        if convergence_threshold is not None:
            if self.method in ['L-BFGS-B', 'TNC', 'SLSQP']:
                options['ftol'] = convergence_threshold
            elif self.method in ['BFGS', 'CG', 'NEWTON-CG']:
                options['gtol'] = convergence_threshold
            elif self.method == 'POWELL':
                options['ftol'] = convergence_threshold
                options['xtol'] = convergence_threshold

        # Run optimization with clean functions
        self._result = minimize(
            fun=objective,
            x0=x0,
            method=self.method,
            jac=gradient,
            callback=callback,
            options=options
        )
        
        # Post-process trajectory after optimization completes
        self._build_trajectory()
        
        return self._package_result()

    def _build_trajectory(self):
        """Build trajectory data from stored points"""
        # Add initial point if not in trajectory
        if len(self.trajectory_points) == 0:
            self.trajectory_points = [self._result.x.copy()]
        
        # Process each trajectory point
        for x in self.trajectory_points:
            # Evaluate all quantities for this point
            unbiased_energy = self.abc_sim.potential.potential(x)
            biased_energy = self.abc_sim.compute_biased_potential(x, deepcopy(unbiased_energy))

            try:
                unbiased_force = -self.abc_sim.potential.gradient(x)
            except NotImplementedError:
                unbiased_force = None
            biased_force = self.abc_sim.compute_biased_force(x, unbiased_force)

            # Store in trajectory
            self.trajectory.append(x.copy())
            self.unbiased_energies.append(unbiased_energy)
            self.biased_energies.append(biased_energy)
            self.unbiased_forces.append(unbiased_force)
            self.biased_forces.append(biased_force)

    def _package_result(self):
        """Package optimization result"""
        result = {
            'x': self._result.x,
            'converged': self._result.success,
            'nit': self._result.nit,
            'message': self._result.message,
            'fun': self._result.fun,
            'nfev': self._result.nfev,
            'njev': getattr(self._result, 'njev', 0)
        }
        
        # Add Hessian information if available
        if hasattr(self._result, 'hess_inv'):
            if self.method == 'BFGS':
                result['hess_inv'] = self._result.hess_inv
            elif self.method == 'L-BFGS-B':
                result['hess_inv'] = self._construct_l_bfgs_hess_inv(self._result.hess_inv)
        
        return result

    def _construct_l_bfgs_hess_inv(self, hess_inv_operator):
        """Convert L-BFGS-B LinearOperator to dense matrix"""
        n = self.abc_sim.dimension
        I = np.eye(n)
        inv_H = np.column_stack([hess_inv_operator.matvec(I[:, i]) for i in range(n)])
        return 0.5 * (inv_H + inv_H.T)  # Symmetrize

    @property
    def inv_hessian(self):
        """Get inverse Hessian approximation if available"""
        if not hasattr(self, '_result'):
            raise AttributeError("Optimization not yet run")
        
        if self.method == 'BFGS' and hasattr(self._result, 'hess_inv'):
            return self._result.hess_inv
        elif self.method == 'L-BFGS-B' and hasattr(self._result, 'hess_inv'):
            return self._construct_l_bfgs_hess_inv(self._result.hess_inv)
        
        raise AttributeError(f"Inverse Hessian not available for method '{self.method}'")


class VanillaSciPyOptimizer:
    """Direct SciPy usage without any wrapper - for comparison"""
    
    def __init__(self, abc_sim, method='BFGS', **kwargs):
        self.abc_sim = abc_sim
        self.method = method.upper()
        self.optimizer_kwargs = kwargs

    def optimize(self, x0, max_steps=None, convergence_threshold=None):
        """Direct SciPy optimization"""
        def objective(x):
            unbiased_energy = self.abc_sim.potential.potential(x)
            biased_energy = self.abc_sim.compute_biased_potential(x, deepcopy(unbiased_energy))
            return biased_energy

        def gradient(x):
            try:
                unbiased_force = -self.abc_sim.potential.gradient(x)
            except NotImplementedError:
                unbiased_force = None
            biased_force = self.abc_sim.compute_biased_force(x, unbiased_force)
            return -biased_force

        options = self.optimizer_kwargs.copy()
        if max_steps is not None:
            options['maxiter'] = max_steps

        if convergence_threshold is not None:
            if self.method in ['L-BFGS-B', 'TNC', 'SLSQP']:
                options['ftol'] = convergence_threshold
            elif self.method in ['BFGS', 'CG', 'NEWTON-CG']:
                options['gtol'] = convergence_threshold

        result = minimize(
            fun=objective,
            x0=x0,
            method=self.method,
            jac=gradient,
            options=options
        )
        
        return result


# Test function to compare approaches
def test_optimization_approaches(abc_sim, x0):
    """Test different optimization approaches to identify the issue"""
    
    print("Testing vanilla SciPy (no wrapper)...")
    vanilla = VanillaSciPyOptimizer(abc_sim, method='BFGS')
    try:
        result1 = vanilla.optimize(x0)
        print(f"✓ Vanilla SciPy: {result1.success}, {result1.message}")
        print(f"  Hessian available: {hasattr(result1, 'hess_inv')}")
    except Exception as e:
        print(f"✗ Vanilla SciPy failed: {e}")
    
    print("\nTesting minimal wrapper...")
    wrapped = ScipyOptimizer(abc_sim, method='BFGS')
    try:
        result2, traj_data = wrapped.descend(x0)
        print(f"✓ Wrapped SciPy: {result2['converged']}, {result2['message']}")
        print(f"  Trajectory length: {len(traj_data['trajectory'])}")
        print(f"  Hessian available: {'hess_inv' in result2}")
    except Exception as e:
        print(f"✗ Wrapped SciPy failed: {e}")
    
    return result1, result2


# Example usage
if __name__ == "__main__":
    # This would test both approaches
    from ca_abc import CurvatureAdaptiveABC
    from potentials import StandardMullerBrown2D
    abc_sim = CurvatureAdaptiveABC(StandardMullerBrown2D(), [0,0])
    result1, result2 = test_optimization_approaches(abc_sim, [0,0])
    # pass