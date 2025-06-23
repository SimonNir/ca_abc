from smart_abc import SmartABC
from optimizers import * 
from potentials import StandardMullerBrown2D
from analysis import ABCAnalysis
import numpy as np

def run_2d_simulation():
    """Run 2D ABC simulation with Muller-Brown potential."""
    np.random.seed(420)
    print("Starting 2D ABC Simulation")
    print("=" * 50)
    
    potential = StandardMullerBrown2D()
    
    abc = SmartABC(
        potential=potential,
        starting_position=[0.0, 0.0],
        perturb_type="adaptive",
        bias_type="adaptive",
        curvature_method="full_hessians",
        default_bias_height=2,
        default_bias_covariance=0.1,
        curvature_bias_height_scale=100,
        default_perturbation_size=0.05,
        max_descent_steps=100, 
        dump_folder=None, 
        descent_convergence_threshold=1e-5,
        curvature_bias_covariance_scale=2,
    )

    
    opt = ScipyOptimizer(abc, 'BFGS')    
    abc.run(max_iterations=100, optimizer=opt, verbose=True)

    # Create analysis and plots
    analyzer = ABCAnalysis(abc)
    analyzer.plot_summary(save_plots=False, filename="2d_smart_abc.png")
    analyzer.plot_diagnostics(save_plots=False, filename="2d_smart_abc_diagnostics.png")

def main():
        run_2d_simulation()
        

if __name__ == "__main__":
        main()