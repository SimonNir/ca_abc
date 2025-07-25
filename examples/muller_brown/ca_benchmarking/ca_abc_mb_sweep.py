import numpy as np
import pandas as pd
import time
from itertools import product
from multiprocessing import Pool
import os
import json
import glob
import ast

from ca_abc.ca_abc import CurvatureAdaptiveABC
from ca_abc.potentials import StandardMullerBrown2D
from ca_abc.optimizers import FIREOptimizer, ScipyOptimizer

# --- Fixed Defaults ---
DEFAULT_PERTURB = 0.01
DEFAULT_BIAS_HEIGHT = 3.8
DEFAULT_BIAS_COV = (0.55 / 5) ** 2

# --- Sweep Parameters ---
PERTURB_TYPES = ['fixed', 'adaptive']
HEIGHT_TYPES = ['fixed', 'adaptive']
COV_TYPES = ['fixed', 'adaptive']

MAX_HEIGHT_FACTORS = [2.0, 3.0, 5.0, 10.0]
MAX_COV_FACTORS = [2.0, 3.0, 5.0, 10.0]

USE_CONSERVATIVE_DELTA = [True, False]
SEEDS = list(range(10))
OPTIMIZERS = [0, 1]  # 0 = FIRE, 1 = Scipy

# --- Paths ---
RESULT_DIR = "ca_abc_mb_results"
FINAL_JSON = "ca_mb_sweep.json"
PAST_CSV = None  # Set to None to disable

# --- Utilities ---

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

# --- Run and Analyze ---

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
    run_id, seed, opt, perturb_type, height_type, cov_type, max_h_factor, max_cov_factor, max_p_factor, use_conservative = args

    np.random.seed(seed)

    try:
        abc = CurvatureAdaptiveABC(
            potential=StandardMullerBrown2D(),
            starting_position=[0.0, 0.0],
            curvature_method="None",
            dump_every=30000,
            dump_folder=f"{RESULT_DIR}/run_{run_id}",

            perturb_type=perturb_type,
            default_perturbation_size=DEFAULT_PERTURB,
            max_perturbation_size=DEFAULT_PERTURB * max_p_factor,
            scale_perturb_by_curvature=False,
            curvature_perturbation_scale=0.0,

            bias_height_type=height_type,
            default_bias_height=DEFAULT_BIAS_HEIGHT,
            max_bias_height=DEFAULT_BIAS_HEIGHT * max_h_factor,
            curvature_bias_height_scale=0.0,

            bias_covariance_type=cov_type,
            default_bias_covariance=DEFAULT_BIAS_COV,
            max_bias_covariance=DEFAULT_BIAS_COV * max_cov_factor,
            curvature_bias_covariance_scale=0.0,

            use_conservative_ems_delta=use_conservative,

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
            'optimizer': 'FIRE' if opt == 0 else 'Scipy',
            'perturb_type': perturb_type,
            'bias_height_type': height_type,
            'bias_covariance_type': cov_type,
            'use_conservative_ems_delta': use_conservative,
            'default_perturbation_size': DEFAULT_PERTURB,
            'default_bias_height': DEFAULT_BIAS_HEIGHT,
            'default_bias_cov': DEFAULT_BIAS_COV,
            'max_perturbation_size': DEFAULT_PERTURB * max_p_factor,
            'max_bias_height': DEFAULT_BIAS_HEIGHT * max_h_factor,
            'max_bias_cov': DEFAULT_BIAS_COV * max_cov_factor,
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

# --- Merging Results ---

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

    dicts = df_json.to_dict(orient="records")
    dicts = [convert_numpy(d) for d in dicts]

    with open(FINAL_JSON, "w") as f:
        json.dump(dicts, f)

    print(f"Final merged results written to: {FINAL_JSON}")
    print(f"Total successful runs: {len(dicts)}")

# --- Entry Point ---

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    completed_csv = get_completed_runs_from_csv(PAST_CSV)
    completed_json = get_completed_runs_from_jsons(RESULT_DIR)
    completed = completed_csv.union(completed_json)

    print(f"Detected {len(completed)} completed runs to skip.")

    sweep = list(product(
        SEEDS,
        OPTIMIZERS,
        PERTURB_TYPES,
        HEIGHT_TYPES,
        COV_TYPES,
        MAX_HEIGHT_FACTORS,
        MAX_COV_FACTORS,
        USE_CONSERVATIVE_DELTA,
    ))

    all_params = [(i, *params) for i, params in enumerate(sweep)]
    todo_params = [p for p in all_params if p[0] not in completed]

    print(f"{len(todo_params)} runs remain to do.")

    nprocs = int(os.environ.get("SLURM_CPUS_PER_TASK", 32))
    start_time = time.time()

    try:
        with Pool(nprocs) as pool:
            pool.map(single_run, todo_params)
        print(f"Sweep complete in {(time.time() - start_time)/60:.2f} minutes")
        merge_results()
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting.")
        os._exit(1)

if __name__ == "__main__":
    main()
