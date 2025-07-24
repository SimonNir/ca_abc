import numpy as np
import pandas as pd
import time
from itertools import product
from multiprocessing import Pool, cpu_count
import os
import json
import glob
import ast

from ca_abc import CurvatureAdaptiveABC
from potentials import StandardMullerBrown2D
from optimizers import FIREOptimizer, ScipyOptimizer

# --- Config ---
RESULT_DIR = "abc_mb_results"
FINAL_JSON = "new_mb_sweep.json"
PAST_CSV = "new_mb_sweep.csv"  # Set to None if no past CSV to check

# --- Helpers ---

def parse_array_list_str(s):
    try:
        return eval(s, {"array": np.array, "np": np})
    except Exception as e:
        print(f"Parse error: {e}")
        return s

def parse_complex_columns(df):
    for col in ['found_minima', 'found_saddles']:
        if col in df.columns:
            df[col] = df[col].apply(parse_array_list_str)

    def safe_parse_list_str(s):
        if pd.isna(s):
            return []
        try:
            return ast.literal_eval(s)
        except Exception:
            return s

    for col in ['energy_calls_at_each_min', 'force_calls_at_each_min']:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_list_str)

    return df

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

# --- Analysis ---

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

# --- Utilities ---

def get_completed_runs_from_csv(csv_path):
    if not csv_path or not os.path.isfile(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        if 'run_id' in df.columns:
            return set(df['run_id'].astype(int).tolist())
        else:
            print(f"run_id column not found in {csv_path}. Ignoring this file.")
            return set()
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        return set()

def get_completed_runs_from_jsons(results_dir):
    completed = set()
    json_paths = glob.glob(os.path.join(results_dir, "run_*.json"))
    for path in json_paths:
        basename = os.path.basename(path)
        try:
            run_id_str = basename.split("_")[1].split(".")[0]
            run_id = int(run_id_str)
            completed.add(run_id)
        except Exception:
            continue
    return completed

# --- Core Execution ---

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
        abc.run(max_iterations=5000, stopping_minima_number=3, optimizer=optimizer, verbose=True, save_summary=False)

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

def merge_results():
    print("Merging individual JSON files into final JSON...")
    all_results = []
    for fname in glob.glob(os.path.join(RESULT_DIR, "run_*.json")):
        try:
            with open(fname) as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            print(f"Failed to load {fname}: {e}")

    df_json = pd.DataFrame(all_results)

    if PAST_CSV and os.path.isfile(PAST_CSV):
        try:
            df_past = pd.read_csv(PAST_CSV)
            df_past = parse_complex_columns(df_past)
            if 'run_id' in df_past.columns:
                df_json = pd.concat([df_json, df_past], ignore_index=True)
                print(f"Merged {len(df_past)} rows from previous CSV.")
            else:
                print(f"run_id column missing from {PAST_CSV}, skipping.")
        except Exception as e:
            print(f"Failed to merge past CSV: {e}")

    # Convert numpy-like objects to Python-native JSON-safe types
    dicts = df_json.to_dict(orient="records")
    dicts = [convert_numpy(d) for d in dicts]

    with open(FINAL_JSON, "w") as f:
        json.dump(dicts, f)

    print(f"Final merged results written to: {FINAL_JSON}")
    print(f"Total successful runs: {len(dicts)}")

# --- Entry Point ---

def main():
    import os
    import time
    from itertools import product
    from multiprocessing import cpu_count

    print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))

    os.makedirs(RESULT_DIR, exist_ok=True)

    completed_from_csv = get_completed_runs_from_csv(PAST_CSV)
    completed_from_json = get_completed_runs_from_jsons(RESULT_DIR)
    completed_runs = completed_from_csv.union(completed_from_json)

    print(f"Detected {len(completed_runs)} completed runs to skip.")

    std_dev_scales = [1/3, 1/5, 1/8, 1/10, 1/14]
    bias_height_fractions = [1/5, 1/10, 1/30, 1/50, 1/100]
    perturbations = [0.55, 0.01, 0.005, 0.001]
    optimizers = [0, 1]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    all_params = list(product(std_dev_scales, bias_height_fractions, perturbations, optimizers, seeds))
    indexed_params = [(i, *params) for i, params in enumerate(all_params)]
    indexed_params = [p for p in indexed_params if p[0] not in completed_runs]

    print(f"{len(indexed_params)} runs remain to do.")

    start_time = time.time()

    try:
        for params in indexed_params:
            single_run(params)
        print(f"Sweep complete in {(time.time() - start_time)/60:.2f} minutes")
        merge_results()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        os._exit(1)  # hard kill immediately

if __name__ == "__main__":
    # parser arguments to determine which batches to run 
    # save each to 1 json at the end, and then combine in your submitter batch script 
    main()
