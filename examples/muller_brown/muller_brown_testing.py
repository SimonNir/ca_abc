from ca_abc import CurvatureAdaptiveABC
from ca_abc.optimizers import * 
from ca_abc.potentials import StandardMullerBrown2D
from ca_abc.analysis import ABCAnalysis
import numpy as np

def run_2d_simulation():
    """Run 2D ABC simulation with Muller-Brown potential."""
    np.random.seed(42)
    print("Starting 2D ABC Simulation")
    print("=" * 50)
    
    abc = CurvatureAdaptiveABC(
        potential=StandardMullerBrown2D(),
        starting_position=np.array([0.6234994049, 0.02803775853]),
        curvature_method="none",
        dump_every=10000,

        perturb_type="fixed",
        default_perturbation_size=0.005,
        scale_perturb_by_curvature=True,
        max_perturbation_size=0.01,
     
        bias_height_type="fixed",
        default_bias_height=.76,
        max_bias_height= 3,

        bias_covariance_type="fixed",
        default_bias_covariance=0.003025,
        max_bias_covariance= 0.02,
        
        use_ema_adaptive_scaling=True,
        conservative_ema_delta=False, 
        
        max_descent_steps=1000,
        descent_convergence_threshold=1e-5, 
        max_acceptable_force_mag=1e99,
    )
    
    opt = FIREOptimizer(abc)  
    abc.run(max_iterations=10000, stopping_minima_number=3, optimizer=opt, verbose=True)

    # Create analysis and plots
    analyzer = ABCAnalysis(abc)
    analyzer.plot_summary(save_plots=False, filename="2d_smart_abc.png", plot_type='both')
    analyzer.plot_diagnostics(save_plots=False, filename="2d_smart_abc_diagnostics.png", plot_type="neither")

def main():
        run_2d_simulation()
        

if __name__ == "__main__":
        main()