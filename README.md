# README

## Overview

This document provides an overview of the components and functionality of the codebase.

---
Eventually, traditional_abc will become deprecated, as we will want smart_abc to include all its functionality and more. We want it so you can indicate whether you want to perturb with noise or softmode, etc. 


## TODO:
- Include "validate config" function to ensure all settings are sensible 
- Likewise, we will move all util-type functions into a dedicated util file. For now, we keep it all together


## Tunable Parameters

This section describes the parameters you can specify to control the behavior of the algorithm. Parameters are either **required** or **optional**, and are set either during **initialization** (`abc = ABC(...)`) or when calling the **`.run()`** method (`abc.run(...)`).

---

### Required Parameters (passed to `__init__`)

These must be specified when creating an `ABC` object.

| Parameter                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `potential`                | A callable representing the potential energy function.                     |
| `expected_barrier_height`  | Estimated average basin depth or barrier height (order-of-magnitude accuracy is sufficient). |

---

### Optional Parameters

These can either be passed to `__init__` or `.run()`, depending on when they are needed.

Setup (passed to `__init__`)

| Parameter                   | Description                                                                                 |
|----------------------------|----------------------------------------------------------------------------------------------|
| `run_mode`                 |`"fastest"` \| `"fast"` \| `"compromise"` \| `"most accurate"` — selects accuracy/performance trade-offs; `"compromise"` by default. <br> **Note**: `"fastest"` runs a traditional ABC |

#### Curvature Estimation (passed to `__init__`)

| Parameter         | Description                                                                                     |
|------------------|-------------------------------------------------------------------------------------------------|
| `curvature_method` | `"full_hessian"` \| `"from_opt"` \| `"lanczos"` \| `"rayleigh"`\| `"ignore"`<br>**Note**: `"from_opt"` requires `optimizer="BFGS"`, `"L-BFGS-B"`, or `"trust-region"`; works best with BFGS. `ignore` runs a traditional ABC |

---

#### Perturbation Strategy (passed to `__init__`)

| Parameter                        | Description                                                                                                 |
|----------------------------------|-------------------------------------------------------------------------------------------------------------|
| `perturb_type`                  | `"random"` \| `"soft_dir"` \| `"dynamic"` — `dynamic` starts with softest mode and switches if needed.      |
| `perturb_dist`                  | `"constant"` \| `"normal"` — determines how perturbation scale is sampled.                                 |                  |
| `scale_perturb_by_curvature`   | If `True`, scale is adjusted by reciprocal curvature. Only allowed for `soft_dir` and `dynamic`.            |
| `default_perturbation_size`    | Used when not scaling by curvature. Default: `0.05`.                                                         |
| `large_perturbation_scale_factor` | Multiplier for what constitutes a “large” jump. Default: `5`.                                             |

---

#### Biasing Strategy (passed to `__init__`)

| Parameter                         | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `bias_type`                      | `"constant"` \| `"smart"`                                                  |
| `default_bias_height`           | Height of the default (constant) bias. Default: `10`.                      |
| `default_bias_covariance`       | Covariance (variance) of the default bias. Default: `1`.                   |
| `curvature_bias_height_scale`   | Multiplies curvature to determine bias height in `smart` mode.             |
| `curvature_bias_covariance_scale` | Multiplies inverse curvature to determine bias covariance in `smart` mode. |

---

#### Descent and Optimization (passed to `__init__`)

| Parameter                     | Description                                                              |
|-------------------------------|--------------------------------------------------------------------------|
| `optimizer`                  | Any supported SciPy optimizer, or `"steepest"` for simple gradient descent. |
| `descent_convergence_threshold` | Threshold for convergence. Default: `1e-5`.                            |
| `max_descent_steps`          | Maximum number of steps allowed per descent. Default: `20`.              |
| `max_descent_step_size`      | Maximum allowed step size. Default: `1`.                                 |

---

#### Run Control (passed to `.run()`)

| Parameter         | Description                                             |
|------------------|---------------------------------------------------------|
| `max_iterations` | Number of iterations to run. Default: `100`.           |
| `starting_position` | Initial position vector.                           |
| `verbose`         | If `True`, prints progress and status updates. Default: `True`. |

---

