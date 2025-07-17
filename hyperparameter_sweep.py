import numpy as np
import pandas as pd
import time

from ca_abc import CurvatureAdaptiveABC
from potentials import StandardMullerBrown2D
from optimizers import FIREOptimizer

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
        'minima_l2_error_mean': np.mean(minima_errors) if minima_errors else np.nan,
        'minima_l2_error_stdv': np.std(minima_errors) if minima_errors else np.nan,
        'minima_l2_error_max': np.max(minima_errors) if minima_errors else np.nan,
        'saddle_l2_error_mean': np.mean(saddle_errors) if saddle_errors else np.nan,
        'saddle_l2_error_stdv': np.std(saddle_errors) if saddle_errors else np.nan,
        'saddle_l2_error_max': np.max(saddle_errors) if saddle_errors else np.nan,
        'bias_count': len(abc.biases),
        'energy_calls_at_each_min': abc.energy_calls_at_each_min if abc.energy_calls_at_each_min else np.nan,
        'force_calls_at_each_min': abc.force_calls_at_each_min if abc.force_calls_at_each_min else np.nan,
    }

def main():
    expected_barrier = 38.0
    expected_length_scale = 0.55

    std_dev_scales = [1/3, 1/5, 1/8, 1/10, 1/14]
    bias_height_fractions = [1/5, 1/10, 1/30, 1/50, 1/100]
    fixed_perturbation = 0.001  # fixed perturbation size

    seeds = [42, 1337, 2024, 7, 314]

    results = []
    run_id = 0

    start_time = time.time()

    for std_scale in std_dev_scales:
        bias_stdv = expected_length_scale * std_scale
        bias_cov = bias_stdv ** 2

        for height_frac in bias_height_fractions:
            bias_height = expected_barrier * height_frac

            for seed in seeds:
                np.random.seed(seed)

                abc = CurvatureAdaptiveABC(
                    potential=StandardMullerBrown2D(),
                    starting_position=[0.0, 0.0],
                    curvature_method="bfgs",

                    perturb_type="fixed",
                    default_perturbation_size=fixed_perturbation,
                    scale_perturb_by_curvature=False,
                    curvature_perturbation_scale=0.0,
                    max_perturbation_size=fixed_perturbation * 5,

                    bias_height_type="fixed",
                    default_bias_height=bias_height,
                    max_bias_height=bias_height * 3,
                    curvature_bias_height_scale=0.0,

                    bias_covariance_type="fixed",
                    default_bias_covariance=bias_cov,
                    max_bias_covariance=bias_cov * 5,
                    curvature_bias_covariance_scale=0.0,

                    max_descent_steps=300,
                    descent_convergence_threshold=1e-6,
                    max_acceptable_force_mag=1e99,
                )

                optimizer = FIREOptimizer(abc)
                abc.run(max_iterations=2000, stopping_minima_number=3, optimizer=optimizer, verbose=False, save_summary=False)

                run_data = analyze_run(abc)
                run_data.update({
                    'run_id': run_id,
                    'seed': seed,
                    'bias_std_dev_scale': std_scale,
                    'bias_covariance': bias_cov,
                    'bias_height_fraction': height_frac,
                    'bias_height': bias_height,
                    'perturbation_size': fixed_perturbation,
                })

                results.append(run_data)
                run_id += 1

                print(f"Completed run {run_id} | seed {seed} | height {bias_height:.4f} | cov {bias_cov:.5f} | perturb {fixed_perturbation:.5f}")

    df = pd.DataFrame(results)
    df.to_csv("muller_brown_fixed_perturb_sweep.csv", index=False)

    print(f"Sweep complete. Total runs: {len(results)}")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()
