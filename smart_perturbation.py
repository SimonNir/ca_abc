import numpy as np
from traditional_abc import GaussianBias, TraditionalABC

class SmartPerturbABC(TraditionalABC):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)

    def reset(self, starting_position=None):
        super().reset(starting_position)
        self.prev_mode = None 

    def compute_hessian_finite_difference(self, position, eps=1e-3):
        """Numerically estimate the Hessian matrix at a point."""
        n = len(position)
        hessian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                delta_i = np.zeros(n)
                delta_j = np.zeros(n)
                delta_i[i] += eps
                delta_j[j] += eps

                f_ij = self.compute_force(position + delta_i + delta_j)
                f_i = self.compute_force(position + delta_i)
                f_j = self.compute_force(position + delta_j)
                f = self.compute_force(position)

                # Use finite difference approximation
                hessian[i, j] = (f_ij[i] - f_i[i] - f_j[i] + f[i]) / (eps ** 2)

        return -hessian  # negative because we're working with -∇V
    
    def compute_exact_softest_hessian_mode(self, position):
        """
        Fully diagonalize the Hessian to find the exact softest mode.
        Only use for low-dimensional potentials.
        """
        hessian = self.compute_hessian_finite_difference(position)
        eigvals, eigvecs = np.linalg.eigh(hessian)

        # Pick direction of lowest curvature (softest mode)
        softest_eigval_idx = np.argmin(eigvals)
        softest_eigval = eigvals[softest_eigval_idx]
        softest_eigvec = eigvecs[:, softest_eigval_idx]

        return softest_eigvec, softest_eigval
    
    def estimate_softest_hessian_mode(self, position, init_direction=None, eps=1e-3, softmode_max_iters=25, tol=1e-4):
        """
        Estimate the softest eigenmode (lowest-curvature direction) at a given position
        without calculating full hessian.
        
        Returns:
            - direction: Unit vector along softest mode (approximate eigenvector)
            - curvature: Approximate eigenvalue along this direction
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
            f1 = self.compute_force(position + eps * direction)
            f2 = self.compute_force(position - eps * direction)

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


    def perturb_along_softest_mode(self, scale, mode):
        softest_direction = mode / np.linalg.norm(mode)
        # Move slightly in that direction
        self.position += scale * softest_direction
        self.trajectory.append(self.position.copy())
        self.forces.append(self.compute_force(self.position))
        self.energies.append(self.total_potential(self.position))
        print(f"Perturbed along softest mode to {self.position}")

    # Overrides traditional
    def run_simulation(
        self,
        max_iterations=30,
        descent_max_steps=100,
        descent_threshold=1e-5,
        perturb_scale=1,
        verbose=True,
        full_hessians=False
    ):
        for iteration in range(max_iterations):
            converged = self.descend(
                max_steps=descent_max_steps,
                convergence_threshold=descent_threshold
            )

            pos = self.position.copy()

            # if converged, perturb away from minimum along softest hessian mode 
            # along pes before new bias deposition
            # supports future implementation of curvature-dependent scaling
            if converged:
                if full_hessians:
                    mode, _ = self.compute_exact_softest_hessian_mode(self.position)
                else:
                    mode, _ = self.estimate_softest_hessian_mode(self.position)
                self.perturb_along_softest_mode(perturb_scale, mode)

            # Deposit bias at minimum
            self.deposit_bias(pos)

            self.store_iter_period()
            
            if verbose:
                print(f"Iteration {iteration+1}/{max_iterations}: "
                      f"Position {self.position}, "
                      f"Total biases: {len(self.bias_list)}")
                      
        print(f"Simulation completed. Total steps: {len(self.trajectory)}")

#####################################

from analysis import plot_results, analyze_basin_visits
from potentials import DoubleWellPotential1D, MullerBrownPotential2D, ComplexPotential1D

def run_1d_simulation():
    """Run 1D ABC simulation with double well potential."""
    np.random.seed(42)
    print("Starting 1D ABC Simulation")
    print("=" * 50)
    
    # potential = DoubleWellPotential1D()
    potential = ComplexPotential1D()
    abc = SmartPerturbABC(
        potential=potential,
        bias_height=1,
        bias_sigma=1,
        basin_radius=0.5,
        starting_position=[-1.2]
    )
    
    abc.run_simulation(max_iterations=20, perturb_scale=1, verbose=True)
    
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total steps: {len(trajectory)}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: {abc.position}")
    
    analyze_basin_visits(abc)
    plot_results(abc, save_plots=True, filename="1d_smart_perturb_abc.png")

def run_2d_simulation():
    """Run 2D ABC simulation with Muller-Brown potential."""
    np.random.seed(420)
    print("Starting 2D ABC Simulation")
    print("=" * 50)
    
    potential = MullerBrownPotential2D()
    abc = SmartPerturbABC(
        potential=potential,
        bias_height=1,
        bias_sigma=1,
        basin_radius=0.5,
        starting_position=[0, 0],
    )
    
    abc.run_simulation(max_iterations=50, perturb_scale=1, full_hessians=True, verbose=True)
    
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total steps: {len(trajectory)}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: {abc.position}")
    
    analyze_basin_visits(abc)
    plot_results(abc, save_plots=True, filename="2d_smart_perturb_abc.png")

def main():
    """Run both 1D and 2D simulations."""
    print("Running 1D Simulation")
    run_1d_simulation()
    
    # print("\nRunning 2D Simulation")
    # run_2d_simulation()

if __name__ == "__main__":
    main()