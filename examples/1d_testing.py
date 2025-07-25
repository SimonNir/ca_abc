from ca_abc import CurvatureAdaptiveABC
from src.ca_abc.optimizers import *
from potentials import Complex1D, DoubleWell1D
from analysis import ABCAnalysis
import numpy as np

def run_1d_simulation():
    """Run 1D ABC simulation with complex potential."""
    np.random.seed(1)

    abc = CurvatureAdaptiveABC(
        potential=Complex1D(),
        starting_position=[0.0],
        curvature_method="bfgs", 
        dump_every=10000,

        perturb_type="fixed",
        default_perturbation_size=0.03,
        scale_perturb_by_curvature=True,
        curvature_perturbation_scale= 0.1, 
        max_perturbation_size= 0.15, 

        bias_height_type="fixed",
        default_bias_height=0.1,
        curvature_bias_height_scale=0.1, 
        max_bias_height=0.25,

        bias_covariance_type="fixed",
        default_bias_covariance=0.01,
        max_bias_covariance=0.015, 
        curvature_bias_covariance_scale=0.005,


        max_descent_steps=1000, 
        descent_convergence_threshold=1e-4
    )

    myopt = FIREOptimizer(abc)
    # myopt = ScipyOptimizer(abc, "BFGS")
    # myopt = ASEOptimizer(abc, optimizer_class='FIRE')
    abc.run(max_iterations=1800, optimizer=myopt, verbose=True)
        
    # Create analysis and plots
    analyzer = ABCAnalysis(abc)
    analyzer.plot_summary(save_plots=True, filename="1d_default_fire.png", plot_type="neither")
    analyzer.plot_diagnostics(save_plots=True, filename="1d_default_fire_diag.png")

    analyzer.create_basin_filling_gif(fps=100, filename="default_1d_filling.gif")

def main():
    run_1d_simulation()

if __name__ == "__main__":
    main()