import numpy as np
from itertools import product
import os, sys
import json
from ca_abc.potentials import StandardMullerBrown2D
from ca_abc.optimizers import FIREOptimizer, ScipyOptimizer
from ca_abc import CurvatureAdaptiveABC

# --- Core Execution ---

def convert_numpy(obj):
    """Recursively convert NumPy objects to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    else:
        return obj


def analyze_run(abc):
    if not abc.saddles:
        abc.summarize()

    return {
        'found_minima': abc.minima,
        'found_saddles': abc.saddles,
        'bias_count': len(abc.bias_list),
        'energy_calls_at_each_min': abc.energy_calls_at_each_min if abc.energy_calls_at_each_min else np.nan,
        'force_calls_at_each_min': abc.force_calls_at_each_min if abc.force_calls_at_each_min else np.nan,
    }

def single_run(args):
    run_id, std_scale, height_frac, perturb, opt, seed = args

    expected_barrier = 38.0
    expected_length_scale = 0.55

    bias_stdv = expected_length_scale * std_scale
    bias_cov = bias_stdv ** 2
    bias_height = expected_barrier * height_frac

    try:
        np.random.seed(seed)

        abc = CurvatureAdaptiveABC(
            potential=StandardMullerBrown2D(),
            starting_position=[0.0, 0.0],
            curvature_method="None",
            dump_every=30000,
            dump_folder=f"{RESULT_DIR}/run_{run_id}",

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
            'bias_std_dev_scale': std_scale,
            'bias_covariance': bias_cov,
            'bias_height_fraction': height_frac,
            'bias_height': bias_height,
            'perturbation_size': perturb,
            'optimizer': 'FIRE' if opt == 0 else 'Scipy'
        })

        run_data = convert_numpy(run_data)

        out_path = os.path.join(RESULT_DIR, f"run_{run_id}.json")
        with open(out_path, "w") as f:
            json.dump(run_data, f)

        print(f"Completed run {run_id} | seed {seed} | optimizer {optimizer.__class__.__name__}")
        return True

    except Exception as e:
        print(f"Run {run_id} failed with error: {e}")
        return False

def get_all_run_params():
    std_dev_scales = [1/3, 1/5, 1/8, 1/10, 1/14]
    bias_height_fractions = [1/5, 1/10, 1/30, 1/50, 1/100]
    perturbations = [0.005]
    optimizers = [0]
    iters = 10
    all_params = list(product(std_dev_scales, bias_height_fractions, perturbations, optimizers, list(range(iters))))
    indexed_params = [(i, *params) for i, params in enumerate(all_params)]
    return indexed_params

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_one.py <run_id>")
        sys.exit(1)

    run_id = int(sys.argv[1])
    os.makedirs(RESULT_DIR, exist_ok=True)

    completed = get_completed_runs_from_jsons(RESULT_DIR)

    if run_id in completed:
        print(f"Run {run_id} already completed. Skipping.")
        return

    params = get_all_run_params()
    param_dict = {i: args for i, *args in params}
    if run_id not in param_dict:
        print(f"Run ID {run_id} not found in parameter list.")
        return

    args = (run_id, *param_dict[run_id])
    success = single_run(args)
    if not success:
        sys.exit(1)  # Non-zero exit for SLURM job error detection

if __name__ == "__main__":
    main()