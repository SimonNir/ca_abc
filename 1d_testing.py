from ca_abc import CurvatureAdaptiveABC
from optimizers import *
from potentials import Complex1D, DoubleWell1D
from analysis import ABCAnalysis
import numpy as np

def run_1d_simulation():
    """Run 1D ABC simulation with complex potential."""
    np.random.seed(1)
    
    potential = Complex1D()
    # potential = DoubleWell1D()

     # abc = CurvatureAdaptiveABC.load_from_disk(potential,
    #     curvature_method="None", 

    #     perturb_type="random",
    #     default_perturbation_size=0.01,
     
    #     bias_height_type="fixed",
    #     default_bias_height=0.1,
    #     min_bias_height= 0.05,
    #     max_bias_height= 0.3,
    #     curvature_bias_height_scale=10,

    #     bias_covariance_type="fixed",
    #     default_bias_covariance=0.001,
    #     min_bias_covariance= 0.0005,
    #     max_bias_covariance= 0.0015,
    #     curvature_bias_covariance_scale=10,
        
    #     max_descent_steps=300, 
    #     descent_convergence_threshold=1e-4
    # )

    abc = CurvatureAdaptiveABC(
        potential=potential,
        starting_position=[0.1],
        curvature_method="bfgs", 
        dump_every=10000,

        perturb_type="adaptive",
        default_perturbation_size=0.05,
        scale_perturb_by_curvature=False,
        random_perturb_every=1000,
     
        bias_height_type="fixed",
        default_bias_height=0.05,
        # min_bias_height= 0.05,
        # max_bias_height= 2,

        bias_covariance_type="adaptive",
        default_bias_covariance=0.003,
        min_bias_covariance= 0.003,
        # max_bias_covariance= 0.2,
        bias_delta = 0.01,
        
        max_descent_steps=100, 
        descent_convergence_threshold=1e-4
    )

    # myopt = SimpleGradientDescent(abc, step_size=0.1)
    myopt = FIREOptimizer(abc)
    # myopt = ScipyOptimizer(abc, "BFGS")
    # myopt = ASEOptimizer(abc, optimizer_class='BFGS')
    abc.run(max_iterations=2200, optimizer=myopt, verbose=False)
        
    # Create analysis and plots
    analyzer = ABCAnalysis(abc)
    analyzer.plot_summary(save_plots=False, filename="1d_smart_abc.png", plot_type="neither")
    analyzer.plot_diagnostics(save_plots=False, filename="1d_smart_abc_diagnostics.png")

    # analyzer.create_basin_filling_gif(fps=60, filename="fixed_bfgs_filling.gif")

    # to hit all 5, 380 iters 8604 energy evals with adaptive
    # without, 
    print(abc.energy_calls)


def main():
    run_1d_simulation()

if __name__ == "__main__":
    main()