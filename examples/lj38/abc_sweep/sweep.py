import numpy as np
import os
import json
import glob
from itertools import product
from ca_abc import CurvatureAdaptiveABC
from ca_abc.potentials import StandardMullerBrown2D
from ca_abc.optimizers import FIREOptimizer, ScipyOptimizer

# --- Config ---
RESULT_DIR = "abc_lj_results"
FINAL_JSON = "lj_sweep_results.json"
ITERS = 5  # Number of iterations per parameter combination

def get_all_run_params():
    # Covariance scales (as fractions of characteristic length scale ~0.55)
    covs = [0.001, 0.005, 0.01, 0.05, 0.1]
    heights = [0.005, 0.01, 0.05, 0.1]
    # Fixed parameters
    perturbations = [0.01]
    optimizers = [0, 1]
    iters = 5
    
    base_params = list(product(covs, heights, 
                            perturbations, optimizers))
    
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
    run_id, cov, height, perturb, opt = args


    try:
        from ca_abc.potentials import CanonicalLennardJonesCluster, cartesian_to_internal, internal_to_cartesian, align_to_canonical
        from ase import io
        # g = io.read("/home/nirenbergsd/ca_abc/examples/lj38/wales_global_min.xyz")
        # g = io.read("/mnt/c/Users/simon/OneDrive - Brown University/Summer 1/ORISE/ca_abc/examples/lj38/wales_global_min.xyz")
        # pos = cartesian_to_internal(align_to_canonical(g.positions.copy()))
        abc = CurvatureAdaptiveABC(
            potential=CanonicalLennardJonesCluster(38),
            # starting_position = pos, 
            curvature_method="None",
            dump_every=30000,
            dump_folder=f"{RESULT_DIR}/run_{run_id}",

            perturb_type="fixed",
            default_perturbation_size=perturb,
            scale_perturb_by_curvature=False,
            max_perturbation_size=perturb * 5,

            bias_height_type="fixed",
            default_bias_height=height,
            # max_bias_height=bias_height * 3,

            bias_covariance_type="fixed",
            default_bias_covariance=cov,
            # max_bias_covariance=bias_cov * 5,
            use_ema_adaptive_scaling=True,
            conservative_ema_delta=False,
            struc_uniqueness_rmsd_threshold=1e-5, 

            max_descent_steps=800,
            descent_convergence_threshold=1e-5 if opt==1 else 1e-2,
            max_acceptable_force_mag=1e99,
        )

        optimizer = FIREOptimizer(abc) if opt == 0 else ScipyOptimizer(abc, "BFGS")
        abc.run(max_iterations=5000, stopping_minima_number=None, optimizer=optimizer, verbose=True)

        run_data = {
            'run_id': run_id,
            'bias_covariance': cov, 
            'bias_height': height, 
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
