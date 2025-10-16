# run_one.py
import sys
import os
from itertools import product
from sweep_v3 import single_run, get_completed_runs_from_csv, get_completed_runs_from_jsons, RESULT_DIR, PAST_CSV

# Replace 'your_script_filename' with the actual .py filename (without .py)

def get_all_run_params():
    std_dev_scales = [1/3, 1/5, 1/8, 1/10, 1/14]
    bias_height_fractions = [1/5, 1/10, 1/30, 1/50, 1/100]
    perturbations = [0.55, 0.01, 0.005, 0.001]
    optimizers = [0, 1]
    seeds = [1,2,3,4,5,6,7,8,9,10]
    all_params = list(product(std_dev_scales, bias_height_fractions, perturbations, optimizers, seeds))
    indexed_params = [(i, *params) for i, params in enumerate(all_params)]
    return indexed_params

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_one.py <run_id>")
        sys.exit(1)

    run_id = int(sys.argv[1])
    os.makedirs(RESULT_DIR, exist_ok=True)

    completed = get_completed_runs_from_csv(PAST_CSV).union(
        get_completed_runs_from_jsons(RESULT_DIR)
    )

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