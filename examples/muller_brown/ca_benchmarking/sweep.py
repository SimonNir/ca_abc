import numpy as np
import os
import json
import glob
from itertools import product
from ca_abc import CurvatureAdaptiveABC
from ca_abc.potentials import StandardMullerBrown2D
from ca_abc.optimizers import FIREOptimizer, ScipyOptimizer

# --- Config ---
RESULT_DIR = "abc_mb_results"
FINAL_JSON = "mb_sweep_results.json"
ITERS = 10  # Number of iterations per parameter combination

def get_all_run_params():
    std_dev_scales = [0.244521]
    bias_height_fractions = [0.02]
    
    # Fixed parameters
    perturbations = [0.005]
    optimizers = [0]

    adaptive_height_scales = np.round(np.linspace(1, 2, num=11), 6).tolist()
    adaptive_cov_scales = np.round(np.linspace(1, 2, num=11), 6).tolist()
    iters = 10
    
    base_params = list(product(std_dev_scales, bias_height_fractions, 
                            perturbations, optimizers, adaptive_height_scales, adaptive_cov_scales))
    
    # Assign unique run_ids
    return [(i, *params) for i, params in enumerate(base_params * iters)]

def convert_numpy(obj):
    """Convert NumPy objects to native Python types for JSON serialization."""
    if isinstance(obj, (np.ndarray, np.integer, np.int64, np.floating, np.float64)):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    elif isinstance(obj, (dict, list)):
        return {k: convert_numpy(v) for k, v in obj.items()} if isinstance(obj, dict) else [convert_numpy(v) for v in obj]
    return obj

def get_completed_runs(results_dir):
    completed = set()
    for path in glob.glob(os.path.join(results_dir, "run_*.json")):
        try:
            run_id = int(os.path.basename(path).split("_")[1].split(".")[0])
            completed.add(run_id)
        except (ValueError, IndexError):
            continue
    return completed

def single_run(args):
    run_id, std_scale, height_frac, perturb, opt, adaptive_height_scale, adaptive_cov_scale = args

    expected_barrier = 38.0
    expected_length_scale = 0.55

    bias_stdv = expected_length_scale * std_scale
    bias_cov = bias_stdv ** 2
    bias_height = expected_barrier * height_frac

    try:
        # No explicit seed setting - using numpy's random state
        abc = CurvatureAdaptiveABC(
            potential=StandardMullerBrown2D(),
            starting_position=np.array([0.6234994049, 0.02803775853]),
            curvature_method="None",
            dump_every=30000,
            dump_folder=f"{RESULT_DIR}/run_{run_id}",

            perturb_type="adaptive",
            default_perturbation_size=perturb,
            scale_perturb_by_curvature=False,

            bias_height_type="adaptive",
            default_bias_height=bias_height,
            min_bias_height=bias_height / adaptive_height_scale,
            max_bias_height=bias_height * adaptive_height_scale,

            bias_covariance_type="adaptive",
            default_bias_covariance=bias_cov,
            min_bias_covariance=bias_cov / adaptive_cov_scale, 
            max_bias_covariance=bias_cov * adaptive_cov_scale,

            use_ema_adaptive_scaling=True,
            conservative_ema_delta=False,

            max_descent_steps=1000,
            descent_convergence_threshold=1e-5,
            max_acceptable_force_mag=1e99,
        )

        optimizer = FIREOptimizer(abc) if opt == 0 else ScipyOptimizer(abc)
        abc.run(max_iterations=5000, stopping_minima_number=3, optimizer=optimizer, verbose=False)

        run_data = {
            'run_id': run_id,
            'bias_std_dev_scale': std_scale,
            'bias_height_fraction': height_frac,
            'perturbation_size': perturb,
            'optimizer': 'FIRE' if opt == 0 else 'Scipy',
            'found_minima': abc.minima,
            'found_saddles': abc.saddles,
            'bias_count': len(abc.bias_list),
            'energy_calls_at_each_min': abc.energy_calls_at_each_min or [],
            'force_calls_at_each_min': abc.force_calls_at_each_min or [],
        }

        with open(os.path.join(RESULT_DIR, f"run_{run_id}.json"), "w") as f:
            json.dump(convert_numpy(run_data), f)

        return True
    except Exception as e:
        print(f"Run {run_id} failed: {str(e)}")
        return False

def merge_results():
    all_results = []
    for fname in glob.glob(os.path.join(RESULT_DIR, "run_*.json")):
        try:
            with open(fname) as f:
                all_results.append(json.load(f))
        except Exception as e:
            print(f"Failed to load {fname}: {e}")

    with open(FINAL_JSON, "w") as f:
        json.dump(all_results, f)
    print(f"Merged {len(all_results)} runs into {FINAL_JSON}")
