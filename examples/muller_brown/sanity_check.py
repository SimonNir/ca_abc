from ca_abc import CurvatureAdaptiveABC
from ca_abc.optimizers import * 
from ca_abc.potentials import StandardMullerBrown2D
from ca_abc.analysis import ABCAnalysis
import numpy as np


def l2_error_to_nearest(point, reference_list):
    return min(np.linalg.norm(np.array(point) - np.array(ref)) for ref in reference_list)

def run_2d_simulation(ad_factor=1.5):
    """Run 2D ABC simulation with Muller-Brown potential."""
    # np.random.seed(420)
    print("Starting 2D ABC Simulation")
    print("=" * 50)

    height = 4.556086
    cov = 0.0088451956
    ad_factor = ad_factor
    bias_type = "adaptive"
    
    abc = CurvatureAdaptiveABC(
        potential=StandardMullerBrown2D(),
        starting_position=np.array([0.6234994049, 0.02803775853]),
        curvature_method="bfgs",
        dump_every=10000,

        perturb_type="fixed",
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
        conservative_ema_delta=True, 
        
        max_descent_steps=1000,
        descent_convergence_threshold=1e-5, 
        max_acceptable_force_mag=1e99,
    )
    
    opt = FIREOptimizer(abc)  
    abc.run(max_iterations=1000, stopping_minima_number=3, optimizer=opt, verbose=False)
    known_saddles = StandardMullerBrown2D().known_saddles()
    avg_err = np.mean([l2_error_to_nearest(saddle, known_saddles) for saddle in abc.saddles])
    avg_calls = abc.force_calls_at_each_min[-1] / len(abc.minima)
    return avg_calls, avg_err


def main():
    np.random.seed(42)
    
    fixed_calls = []
    fixed_errs = []
    ad_calls = []
    ad_errs = []
    for i in range(10):
        fixed_c, fixed_e = run_2d_simulation(ad_factor=1)
        fixed_calls.append(fixed_c)
        fixed_errs.append(fixed_e)

        ad_c, ad_e = run_2d_simulation(ad_factor=2.5)
        ad_calls.append(ad_c)
        ad_errs.append(ad_e)
    
    print(f"Mean Fixed: ({np.mean(fixed_calls)}, {np.mean(fixed_errs)})")
    print(f"Mean Ad: ({np.mean(ad_calls)}, {np.mean(ad_errs)})")
        

if __name__ == "__main__":
        main()