import numpy as np
from potentials import DoubleWellPotential1D, MullerBrownPotential2D

from traditional_abc import GaussianBias, TraditionalABC

class SmartPerturbABC(TraditionalABC):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)

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


    def perturb_along_softest_mode(self, scale, hessian):
        eigvals, eigvecs = np.linalg.eigh(hessian)

        # Pick direction of lowest curvature (softest mode)
        softest_eigvec = eigvecs[:, np.argmin(eigvals)]

        softest_direction = softest_eigvec / np.linalg.norm(softest_eigvec)
        print(self.position)
        # Move slightly in that direction
        self.position += scale * softest_direction
        print(self.position)
        print(f"Perturbed along softest mode to {self.position}")

    def run_simulation(
        self,
        max_iterations=30,
        descent_max_steps=100,
        descent_threshold=1e-5,
        perturb_scale=1,
        verbose=True
    ):
        for iteration in range(max_iterations):
            converged = self.descend(
                max_steps=descent_max_steps,
                convergence_threshold=descent_threshold
            )

            hessian_before_new_bias = self.compute_hessian_finite_difference(self.position)

            self.deposit_bias()

            if converged:
                self.perturb_along_softest_mode(perturb_scale, hessian_before_new_bias)
            
            if verbose:
                print(f"Iteration {iteration+1}/{max_iterations}: "
                      f"Position {self.position}, "
                      f"Total biases: {len(self.bias_list)}")
                      
        print(f"Simulation completed. Total steps: {len(self.trajectory)}")

from analysis import plot_results, analyze_basin_visits

def run_1d_simulation():
    """Run 1D ABC simulation with double well potential."""
    np.random.seed(42)
    print("Starting 1D ABC Simulation")
    print("=" * 50)
    
    potential = DoubleWellPotential1D()
    abc = SmartPerturbABC(
        potential=potential,
        bias_height=2,
        bias_sigma=0.5,
        basin_radius=0.5,
        starting_position=[-1.2]
    )
    
    abc.run_simulation(max_iterations=8, verbose=True)
    
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
        bias_height=20,
        bias_sigma=1,
        basin_radius=0.5,
        starting_position=[0, 0]
    )
    
    abc.run_simulation(max_iterations=3, verbose=True)
    
    trajectory = abc.get_trajectory()
    print(f"\nSimulation Summary:")
    print(f"Total steps: {len(trajectory)}")
    print(f"Biases deposited: {len(abc.bias_list)}")
    print(f"Final position: {abc.position}")
    
    analyze_basin_visits(abc)
    plot_results(abc, save_plots=True, filename="2d_smart_perturb_abc.png")

def main():
    """Run both 1D and 2D simulations."""
    # print("Running 1D Simulation")
    # run_1d_simulation()
    
    print("\nRunning 2D Simulation")
    run_2d_simulation()

if __name__ == "__main__":
    main()