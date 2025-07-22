import numpy as np
import pandas as pd
import time
from itertools import product
from multiprocessing import Pool, cpu_count, Lock, Manager
import os
import csv

from ca_abc import CurvatureAdaptiveABC
from potentials import StandardMullerBrown2D
from optimizers import FIREOptimizer, ScipyOptimizer

# Shared CSV file name
CSV_FILE = "new_mb_sweep.csv"

def analyze_run(abc):
    known_minima = abc.potential.known_minima()
    known_saddles = abc.potential.known_saddles()
    abc.summarize()

    def l2_error_to_nearest(point, reference_list):
        return min(np.linalg.norm(np.array(point) - np.array(ref)) for ref in reference_list)

    minima_errors = [l2_error_to_nearest(min_, known_minima) for min_ in abc.minima]
    saddle_errors = [l2_error_to_nearest(sad, known_saddles) for sad in abc.saddles]

    return {
        'found_minima': abc.minima,
        'found_saddles': abc.saddles,
        'bias_count': len(abc.bias_list),
        'energy_calls_at_each_min': abc.energy_calls_at_each_min if abc.energy_calls_at_each_min else np.nan,
        'force_calls_at_each_min': abc.force_calls_at_each_min if abc.force_calls_at_each_min else np.nan,
    }

def single_run(args):
    run_id, std_scale, height_frac, perturb, opt, seed, lock = args

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

        with lock:
            df = pd.DataFrame([run_data])
            file_exists = os.path.isfile(CSV_FILE)
            df.to_csv(CSV_FILE, mode='a', index=False, header=not file_exists)

        print(f"Completed run {run_id} | seed {seed} | height {bias_height:.4f} | cov {bias_cov:.5f} | perturb {perturb:.5f} | optimizer {optimizer.__class__.__name__}")
        return True

    except Exception as e:
        print(f"Run {run_id} failed with error: {e}")
        return False

def main():
    std_dev_scales = [1/3, 1/5, 1/8, 1/10, 1/14]
    bias_height_fractions = [1/5, 1/10, 1/30, 1/50, 1/100]
    perturbations = [0.55, 0.01, 0.005, 0.001]
    optimizers = [0, 1]  # 0 = FIRE, 1 = Scipy
    seeds = [1,2,3,4,5,6,7,8,9,10]  # or multiple seeds

    all_params = list(product(std_dev_scales, bias_height_fractions, perturbations, optimizers, seeds))
    indexed_params = [(i, *params) for i, params in enumerate(all_params)]

    start_time = time.time()

    nprocs = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))

    with Manager() as manager:
        lock = manager.Lock()
        args_with_lock = [(run_id, s, h, p, o, seed, lock) for (run_id, s, h, p, o, seed) in indexed_params]

        with Pool(nprocs) as pool:
            pool.map(single_run, args_with_lock)

    print(f"Parallel sweep complete.")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()
