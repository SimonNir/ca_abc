from ca_abc import CurvatureAdaptiveABC
from optimizers import *
from potentials import Complex1D, DoubleWell1D
from analysis import ABCAnalysis
import numpy as np

def run_1d_simulation():
    """Run 1D ABC simulation with complex potential."""
    np.random.seed(1)
    
    potential = Complex1D()

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
        starting_position=[0.0],
        curvature_method="finite_diff", 
        dump_every=1000,

        perturb_type="fixed",
        default_perturbation_size=0.05,
        scale_perturb_by_curvature=False,
        random_perturb_every=3,
     
        bias_height_type="fixed",
        default_bias_height=0.1,
        min_bias_height= 0.05,
        max_bias_height= 4,
        curvature_bias_height_scale=0.02,

        bias_covariance_type="fixed",
        default_bias_covariance=0.01,
        min_bias_covariance= 0.01/10,
        max_bias_covariance= 0.01*20,
        curvature_bias_covariance_scale=1/6,
        
        max_descent_steps=100, 
        descent_convergence_threshold=1e-4
    )

    # myopt = SimpleGradientDescent(abc, step_size=0.01)
    myopt = ScipyOptimizer(abc, method="L-BFGS-B")
    # myopt = ASEOptimizer(abc, optimizer_class='FIRE')
    abc.run(max_iterations=400, optimizer=myopt, verbose=True)
        
    # Create analysis and plots
    analyzer = ABCAnalysis(abc)
    analyzer.plot_summary(save_plots=False, filename="1d_smart_abc.png", plot_type="neither")
    analyzer.plot_diagnostics(save_plots=False, filename="1d_smart_abc_diagnostics.png")

    analyzer.create_basin_filling_gif(fps=60, filename="testing_ase_opt.gif")


def main():
    run_1d_simulation()

if __name__ == "__main__":
    main()