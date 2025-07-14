from ca_abc import CurvatureAdaptiveABC
from optimizers import * 
from potentials import StandardMullerBrown2D
from analysis import ABCAnalysis
import numpy as np

def run_2d_simulation():
    """Run 2D ABC simulation with Muller-Brown potential."""
    np.random.seed(402)
    print("Starting 2D ABC Simulation")
    print("=" * 50)
    
    potential = StandardMullerBrown2D()
    
    abc = CurvatureAdaptiveABC(
        potential=potential,
        starting_position=[0.0, 0.0],
        curvature_method="bfgs",

        perturb_type="adaptive",
        default_perturbation_size=0.01,
        scale_perturb_by_curvature=False,
        curvature_perturbation_scale=1,
     
        bias_height_type="fixed",
        default_bias_height=2,
        # min_bias_height= 0.5,
        # max_bias_height= 3,
        # curvature_bias_height_scale=100,

        bias_covariance_type="adaptive",
        default_bias_covariance=0.01,
        min_bias_covariance= 0.01,
        max_bias_covariance= 0.05,
        bias_delta=0.00002,
        # curvature_bias_covariance_scale=10,
        
        max_descent_steps=300,
        descent_convergence_threshold=1e-6, 
        max_acceptable_force_mag=1e99,
    )
    
    opt = ScipyOptimizer(abc, 'BFGS')  
    # opt = FIREOptimizer(abc)  
    # opt = SimpleGradientDescent(abc, 1e-6)
    abc.run(max_iterations=350, optimizer=opt, verbose=True)

    # Create analysis and plots
    analyzer = ABCAnalysis(abc)
    analyzer.plot_summary(save_plots=False, filename="2d_smart_abc.png", plot_type='both')
    analyzer.plot_diagnostics(save_plots=False, filename="2d_smart_abc_diagnostics.png", plot_type="neither")

    # 39 iter, 586 calls to jump to the second minimum w random perturb
    # with softmode, 41 iter and 565 force calls 
    print(abc.energy_calls)
    # print(abc.force_calls)

def main():
        run_2d_simulation()
        

if __name__ == "__main__":
        main()