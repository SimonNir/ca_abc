import numpy as np
import pandas as pd
import os
import json
import glob
import ast 
from sweep import get_run_params, get_all_run_params

# --- Config ---
RESULT_DIR = "abc_mb_results"
FINAL_JSON = "fine_mb_sweep.json"
PAST_CSV = None  # Set to None if no past CSV to check

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

def merge_results():
    print("Merging individual JSON files into final JSON...")
    all_results = []

    files = glob.glob(os.path.join(RESULT_DIR, "run_*.json"))
    print(f"Found {len(files)} result files.")

    for fname in files:
        try:
            with open(fname) as f:
                data = json.load(f)

                # if "run_id" not in data:
                #     print(f"Skipping {fname}: no run_id found.")
                #     continue
                # # Originally had this block because forgot to include in the original jsons at first
                # try:
                #     params = get_run_params(data["run_id"])
                #     data["adaptive_height_scale"] = params[4]
                #     data["adaptive_cov_scale"] = params[5]
                # except Exception as e:
                #     print(f"run_id = {data['run_id']}, param injection failed: {e}")
                
                all_results.append(data)

        except Exception as e:
            print(f"Failed to load {fname}: {e}")

    if not all_results:
        print("No successful results loaded.")
        return

    print(f"Loaded {len(all_results)} runs.")

    df_json = pd.DataFrame(all_results)
    print(f"Resulting DataFrame shape: {df_json.shape}")

    # Optionally, check what's inside
    print("Sample rows:")
    print(df_json.head())

    # Final serialization
    dicts = df_json.to_dict(orient="records")
    dicts = [convert_numpy(d) for d in dicts]

    with open(FINAL_JSON, "w") as f:
        json.dump(dicts, f)

    print(f"Final merged results written to: {FINAL_JSON}")
    print(f"Total successful runs: {len(dicts)}")



if __name__ == "__main__":
    merge_results()