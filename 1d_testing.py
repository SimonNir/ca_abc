from ca_abc import CurvatureAdaptiveABC
from optimizers import *
from potentials import Complex1D, DoubleWell1D
from analysis import ABCAnalysis
import numpy as np

def run_1d_simulation():
    """Run 1D ABC simulation with complex potential."""
    np.random.seed(1)
    
    potential = Complex1D()
    potential = DoubleWell1D()

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

        perturb_type="fixed",
        default_perturbation_size=0.05,
        scale_perturb_by_curvature=False,
        random_perturb_every=1000,
     
        bias_height_type="adaptive",
        default_bias_height=0.3,
        min_bias_height= 0.05,
        max_bias_height= 2,
        curvature_bias_height_scale=1,

        bias_covariance_type="fixed",
        default_bias_covariance=0.005,
        min_bias_covariance= 0.05/10,
        max_bias_covariance= 0.05*10,
        curvature_bias_covariance_scale=1,
        
        max_descent_steps=100, 
        descent_convergence_threshold=1e-4
    )

    # myopt = SimpleGradientDescent(abc, step_size=0.1)
    # myopt = FIREOptimizer(abc)
    myopt = ScipyOptimizer(abc, "BFGS")
    # myopt = ASEOptimizer(abc, optimizer_class='BFGS')
    abc.run(max_iterations=1, optimizer=myopt, verbose=True)
        
    # Create analysis and plots
    analyzer = ABCAnalysis(abc)
    analyzer.plot_summary(save_plots=False, filename="1d_smart_abc.png", plot_type="neither")
    # analyzer.plot_diagnostics(save_plots=False, filename="1d_smart_abc_diagnostics.png")

    # analyzer.create_basin_filling_gif(fps=60, filename="double_well_filling.gif")


def main():
    run_1d_simulation()

if __name__ == "__main__":
    main()