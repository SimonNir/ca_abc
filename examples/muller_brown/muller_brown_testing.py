from ca_abc import CurvatureAdaptiveABC
from ca_abc.optimizers import * 
from ca_abc.potentials import StandardMullerBrown2D
from ca_abc.analysis import ABCAnalysis
import numpy as np

def run_2d_simulation():
    """Run 2D ABC simulation with Muller-Brown potential."""
    # np.random.seed(420)
    print("Starting 2D ABC Simulation")
    print("=" * 50)

    # height = 4.556086
    height = 2
    # cov = 0.0088451956
    cov = 0.002
    ad_factor = 1.5
    bias_type = "fixed"
    
    abc = CurvatureAdaptiveABC(
        potential=StandardMullerBrown2D(),
        starting_position=np.array([0.6234994049, 0.02803775853]),
        curvature_method="bfgs",
        dump_every=10000,

        perturb_type="stochastic",
        default_perturbation_size=0.001,
        scale_perturb_by_curvature=False,
        # min_perturbation_size=0.005/1.5,
        max_perturbation_size=0.005*1.5,
     
        bias_height_type=bias_type,
        default_bias_height=height,
        min_bias_height= height/ad_factor,
        max_bias_height=height*ad_factor, 

        bias_covariance_type=bias_type,
        # default_bias_covariance=0.003025,
        default_bias_covariance=cov,
        min_bias_covariance= cov/ad_factor,
        max_bias_covariance= cov*ad_factor,
        
        use_ema_adaptive_scaling=True,
        conservative_ema_delta=False, 
        
        max_descent_steps=1000,
        descent_convergence_threshold=1e-5, 
        max_acceptable_force_mag=1e99,
        min_descent_steps=0,
    )
    
    # opt = ScipyOptimizer(abc)  
    opt = FIREOptimizer(abc, max_step_size=None) #0.05
    abc.run(max_iterations=8000, stopping_minima_number=3, optimizer=opt, verbose=False, verbose_opt=False)

    # Create analysis and plots
    analyzer = ABCAnalysis(abc)
    analyzer.plot_summary(save_plots=False, filename="2d_smart_abc.png", plot_type='both')
    analyzer.plot_diagnostics(save_plots=False, filename="2d_smart_abc_diagnostics.png", plot_type="neither")

def main():
        run_2d_simulation()
        

if __name__ == "__main__":
        main()