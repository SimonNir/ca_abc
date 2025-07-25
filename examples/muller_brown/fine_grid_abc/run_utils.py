import numpy as np, os, json
from ca_abc.potentials import StandardMullerBrown2D
from ca_abc.optimizers import FIREOptimizer, ScipyOptimizer
from ca_abc import CurvatureAdaptiveABC

def convert_numpy(obj):
    if isinstance(obj, dict): return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    return obj

def analyze_run(abc):
    if not abc.saddles: abc.summarize()
    return {
        'found_minima': abc.minima,
        'found_saddles': abc.saddles,
        'bias_count': len(abc.bias_list),
        'energy_calls_at_each_min': abc.energy_calls_at_each_min or np.nan,
        'force_calls_at_each_min': abc.force_calls_at_each_min or np.nan,
    }

def single_run(args):
    run_id, std, height, perturb, opt, seed = args
    np.random.seed(seed)
    barrier, length = 38.0, 0.55
    bias_std = length * std
    bias_cov = bias_std ** 2
    bias_height = barrier * height

    abc = CurvatureAdaptiveABC(
        potential=StandardMullerBrown2D(),
        starting_position=[0.0, 0.0],
        curvature_method="None",
        dump_every=30000,
        dump_folder=None,

        perturb_type="fixed",
        default_perturbation_size=perturb,
        scale_perturb_by_curvature=False,
        curvature_perturbation_scale=0.0,
        max_perturbation_size=perturb * 5,

        bias_height_type="fixed",
        default_bias_height=bias_height,
        max_bias_height=bias_height * 3,
        curvature_bias_height_scale=0.0,

        bias_covariance_type="fixed",
        default_bias_covariance=bias_cov,
        max_bias_covariance=bias_cov * 5,
        curvature_bias_covariance_scale=0.0,

        max_descent_steps=1000,
        descent_convergence_threshold=1e-5,
        max_acceptable_force_mag=1e99,
    )

    optimizer = FIREOptimizer(abc) if opt == 0 else ScipyOptimizer(abc)
    abc.run(max_iterations=5000, stopping_minima_number=3, optimizer=optimizer, verbose=False, save_summary=False)

    run_data = analyze_run(abc)
    run_data.update({
        'run_id': run_id,
        'seed': seed,
        'bias_std_dev_scale': std,
        'bias_covariance': bias_cov,
        'bias_height_fraction': height,
        'bias_height': bias_height,
        'perturbation_size': perturb,
        'optimizer': 'FIRE' if opt == 0 else 'Scipy'
    })

    return run_data
