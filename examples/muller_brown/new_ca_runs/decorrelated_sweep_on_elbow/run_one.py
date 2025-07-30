import sys
import os
import json
from sweep import single_run, get_all_run_params, get_completed_runs, RESULT_DIR
import gc 

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_one.py <run_id1> <run_id2> ...")
        sys.exit(1)

    os.makedirs(RESULT_DIR, exist_ok=True)
    run_ids = list(map(int, sys.argv[1:]))
    all_params = {run_id: params for run_id, *params in get_all_run_params()}

    for run_id in run_ids:
        if run_id not in all_params:
            print(f"Invalid run_id: {run_id}")
            continue

        print(f"Starting run {run_id}")
        success = single_run((run_id, *all_params[run_id]))
        if not success:
            print(f"Warning: Run {run_id} failed")
        gc.collect()

if __name__ == "__main__":
    main()