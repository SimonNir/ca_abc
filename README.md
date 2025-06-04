# README

## Overview

This document provides an overview of the components and functionality of the codebase.

> **Note:** The `traditional_abc` module will eventually be deprecated. The goal is for `smart_abc` to subsume all its features and add more flexibility, such as choosing between noise or softmode perturbations.

---

## TODO

- Add a "validate config" function to ensure all settings are sensible.
- Move all utility functions into a dedicated `util` file (currently kept together for convenience).

---

## Tunable Parameters

Parameters control the algorithm's behavior. They are either **required** or **optional**, and are set during **initialization** (`abc = ABC(...)`) or when calling **`.run()`** (`abc.run(...)`).

---

### Required Parameters (`__init__`)

| Parameter                | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `potential`              | Callable representing the potential energy function.                        |
| `expected_barrier_height`| Estimated average basin depth/barrier height (order-of-magnitude accuracy is sufficient). |
'dump_folder' - set to None for quick calculations; otherwise, automatically generates folder to store run
dump_every - number of iterations to dump between - if set to 0, will never dump or clear RAM 
---

### Optional Parameters

Optional parameters can be passed to `__init__` or `.run()` as needed.

#### Setup (`__init__`)

| Parameter      | Description                                                                                                              |
|----------------|--------------------------------------------------------------------------------------------------------------------------|
| `run_mode`     | `"fastest"` \| `"fast"` \| `"compromise"` \| `"most accurate"` — selects accuracy/performance trade-offs. Default: `"compromise"`. <br> **Note:** `"fastest"` runs a traditional ABC. |
| `starting_position` | Initial position vector.                                                                                           |
| `dump_interval`     | Interval for saving progress.                                                                                      |

---

#### Curvature Estimation (`__init__`)

| Parameter           | Description                                                                                                         |
|---------------------|---------------------------------------------------------------------------------------------------------------------|
| `curvature_method`  | `"full_hessian"` \| `"from_opt"` \| `"lanczos"` \| `"rayleigh"` \| `"ignore"` <br> **Note:** `"from_opt"` requires `optimizer="BFGS"`, `"L-BFGS-B"`, or `"trust-region"`; works best with BFGS. `"ignore"` runs a traditional ABC. |

---

#### Perturbation Strategy (`__init__`)

| Parameter                      | Description                                                                                             |
|--------------------------------|---------------------------------------------------------------------------------------------------------|
| `perturb_type`                 | `"random"` \| `"soft_dir"` \| `"dynamic"` — `dynamic` starts with softest mode and switches if needed.  |
| `perturb_dist`                 | `"constant"` \| `"normal"` — determines how perturbation scale is sampled.                              |
| `scale_perturb_by_curvature`   | If `True`, scale is adjusted by reciprocal curvature (only for `soft_dir` and `dynamic`).               |
| `default_perturbation_size`    | Used when not scaling by curvature. Default: `0.05`.                                                    |
| `large_perturbation_scale_factor` | Multiplier for what constitutes a “large” jump. Default: `5`.                                         |

---

#### Biasing Strategy (`__init__`)

| Parameter                        | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `bias_type`                      | `"constant"` \| `"smart"`                                                  |
| `default_bias_height`             | Height of the default (constant) bias. Default: `10`.                      |
| `default_bias_covariance`         | Covariance (variance) of the default bias. Default: `1`.                   |
| `curvature_bias_height_scale`     | Multiplies curvature to determine bias height in `smart` mode.             |
| `curvature_bias_covariance_scale` | Multiplies inverse curvature to determine bias covariance in `smart` mode. |

---

#### Descent and Optimization (`__init__`)

| Parameter                     | Description                                                              |
|-------------------------------|--------------------------------------------------------------------------|
| `optimizer`                   | Any supported SciPy optimizer, or `"steepest"` for simple gradient descent. |
| `descent_convergence_threshold` | Threshold for convergence. Default: `1e-5`.                            |
| `max_descent_steps`           | Maximum number of steps per descent. Default: `20`.                      |
| `max_descent_step_size`       | Maximum allowed step size. Default: `1`.                                 |

---

#### Run Control (`.run()`)

| Parameter           | Description                                                        |
|---------------------|--------------------------------------------------------------------|
| `max_iterations`    | Number of iterations to run. Default: `100`.                       |
| `starting_position` | Initial position vector.                                           |
| `verbose`           | If `True`, prints progress and status updates. Default: `True`.    |
| `max_steps`         | Maximum number of steps allowed.                                   |
| `stopping_condition`| `"minima"` \| `"abc_iters"` \| `"steps"` \| `"force_calls"`        |

---

## File Structure

```text
abc_runs/
├── run_YYYYMMDD/                # Run-specific folder
│   ├── history_000000-000123.h5 # Trajectory/energy/force chunks
│   ├── history_000124-000256.h5 # (Next chunk)
│   ├── biases.h5                # Growing list of all biases
│   └── landmarks.h5             # Minima/saddles found
```