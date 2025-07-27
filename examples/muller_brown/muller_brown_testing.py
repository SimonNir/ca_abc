from ca_abc import CurvatureAdaptiveABC
from ca_abc.optimizers import * 
from ca_abc.potentials import StandardMullerBrown2D
from ca_abc.analysis import ABCAnalysis
import numpy as np

def run_2d_simulation():
    """Run 2D ABC simulation with Muller-Brown potential."""
    np.random.seed(420)
    print("Starting 2D ABC Simulation")
    print("=" * 50)
    
    abc = CurvatureAdaptiveABC(
        potential=StandardMullerBrown2D(),
        starting_position=np.array([0.6234994049, 0.02803775853]),
        curvature_method="bfgs",
        dump_every=10000,

        perturb_type="adaptive",
        default_perturbation_size=0.005,
        scale_perturb_by_curvature=False,
        # min_perturbation_size=0.005/1.5,
        max_perturbation_size=0.005*1.5,
     
        bias_height_type="adaptive",
        default_bias_height=.76,
        min_bias_height=.76/1.5,
        max_bias_height= 1.5*.76,

        bias_covariance_type="adaptive",
        # default_bias_covariance=0.003025,
        default_bias_covariance=0.018087,
        min_bias_covariance= 0.018087/1.5,
        max_bias_covariance= 1.5*0.018087,
        
        use_ema_adaptive_scaling=True,
        conservative_ema_delta=False, 
        
        max_descent_steps=1000,
        descent_convergence_threshold=1e-5, 
        max_acceptable_force_mag=1e99,
    )
    
    opt = FIREOptimizer(abc)  
    abc.run(max_iterations=400, stopping_minima_number=3, optimizer=opt, verbose=True)

    # Create analysis and plots
    analyzer = ABCAnalysis(abc)
    # analyzer.plot_summary(save_plots=False, filename="2d_smart_abc.png", plot_type='both')
    # analyzer.plot_diagnostics(save_plots=False, filename="2d_smart_abc_diagnostics.png", plot_type="neither")

    # fixed: 52002
    # adaptive cov: 41283
    # adaptive cov + height: 32335
    # adaptive cov + height + unscaled perturb:27850
    # adaptive cov + height + scaled perturb: 30205

def main():
        run_2d_simulation()
        

if __name__ == "__main__":
        main()