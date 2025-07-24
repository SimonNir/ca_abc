import numpy as np
import pandas as pd
import os
import json
import glob

# --- Config ---
RESULT_DIR = "abc_mb_results"
FINAL_JSON = "new_mb_sweep.json"
PAST_CSV = "new_mb_sweep.csv"  # Set to None if no past CSV to check

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


if __name__ == "__main__":
    # parser arguments to determine which batches to run 
    # save each to 1 json at the end, and then combine in your submitter batch script 
    merge_results()