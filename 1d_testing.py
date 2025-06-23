from smart_abc import SmartABC
from optimizers import *
from potentials import Complex1D, DoubleWell1D
from analysis import ABCAnalysis
import numpy as np

def run_1d_simulation():
    """Run 1D ABC simulation with complex potential."""
    np.random.seed(1)
    
    potential = Complex1D()

    abc = SmartABC(
        potential=potential,
        starting_position=[0.0],
        default_bias_height=2,
        default_bias_covariance=0.2,
        default_perturbation_size=0.01,
        perturb_type="random",
        bias_type="smart",
        expected_barrier_height=2,

        # curvature_bias_covariance_scale=1e-6,
        curvature_method="full_hessian",
        # curvature_method="estimate",
        dump_folder=None, 
        dump_every=0,
        max_descent_steps=100,
        descent_convergence_threshold=1e-5
    )
    
    opt = ScipyOptimizer(abc, method="BFGS")
    # opt = ScipyOptimizer(abc, method='trust-krylov', initial_trust_radius=1., max_trust_radius=1000)
    # opt = ASEOptimizer(abc, optimizer_class="FIRE")
    # opt = ConservativeSteepestDescent(abc)

    abc.run(max_iterations=30, optimizer=opt, verbose=True)
        
    # Create analysis and plots
    analyzer = ABCAnalysis(abc)
    analyzer.plot_summary(save_plots=False, filename="1d_smart_abc.png")
    # analyzer.plot_diagnostics(save_plots=False, filename="1d_smart_abc_diagnostics.png")

def main():
    run_1d_simulation()

if __name__ == "__main__":
    main()