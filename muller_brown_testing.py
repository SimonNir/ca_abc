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
    
    abc = CurvatureAdaptiveABC(
        potential=StandardMullerBrown2D(),
        starting_position=[0.0, 0.0],
        curvature_method="None",
        dump_every=10000

        perturb_type="fixed",
        default_perturbation_size=0.001,
        scale_perturb_by_curvature=False,
        curvature_perturbation_scale=0.1,
        max_perturbation_size=0.01,
     
        bias_height_type="fixed",
        default_bias_height=1,
        max_bias_height= 3,
        curvature_bias_height_scale=100,

        bias_covariance_type="fixed",
        default_bias_covariance=0.001,
        max_bias_covariance= 0.02,
        curvature_bias_covariance_scale=1,
        
        use_ema_adaptive_scaling=True,
        
        max_descent_steps=500,
        descent_convergence_threshold=1e-6, 
        max_acceptable_force_mag=1e99,
    )
    
    opt = FIREOptimizer(abc)  
    abc.run(max_iterations=5000, stopping_minima_number=3, optimizer=opt, verbose=True)

    # Create analysis and plots
    analyzer = ABCAnalysis(abc)
    analyzer.plot_summary(save_plots=False, filename="2d_smart_abc.png", plot_type='both')
    analyzer.plot_diagnostics(save_plots=False, filename="2d_smart_abc_diagnostics.png", plot_type="neither")

    print(abc.saddles)

def main():
        run_2d_simulation()
        

if __name__ == "__main__":
        main()