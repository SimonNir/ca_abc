# Curvature-Adaptive Autonomous Basin Climbing (CA-ABC)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/ca-abc.svg?logo=pypi)](https://pypi.org/project/ca-abc/) <!-- Placeholder -->

A Python implementation of the **Autonomous Basin Climbing (ABC)** algorithm for the efficient, on-the-fly exploration of complex potential energy surfaces (PES).

This package is designed for discovering reaction pathways, transition states, and local minima in chemistry, physics, and materials science simulations without requiring predefined collective variables or reaction endpoints. To our knowledge, this is the only publicly available Python implementation of the ABC algorithm.



---

## Key Features

-   **Automated Discovery**: Combines minima discovery and transition state identification into a single, autonomous workflow.
-   **Curvature-Adaptive Biasing**: Uses local Hessian information (from BFGS or other methods) to shape anisotropic Gaussian biases, effectively filling basins and encouraging escapes.
-   **Intelligent Perturbations**: Guides escapes along soft-mode directions (low-curvature pathways) for physically meaningful transitions.
-   **Flexible Exploration Modes**: Supports fully **deterministic**, **randomized-adaptive**, and **stochastic** exploration strategies.
-   **ASE Integration**: Works seamlessly with the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) for advanced atomistic simulations.
-   **Subspace Biasing**: Easily restrict biasing and perturbations to specific atoms or degrees of freedom using `biased_atom_indices`.

---

## Installation

First, clone the repository and navigate into the directory:
```bash
git clone [https://github.com/SimonNir/ca-abc.git](https://github.com/SimonNir/ca-abc.git)
cd ca_abc
````

Then, install the package in editable mode with its dependencies:

```bash
pip install -e .
```

#### Dependencies

This project requires:

  * Python 3.7+
  * NumPy
  * SciPy
  * Atomic Simulation Environment (ASE)
  * Matplotlib (for analysis tools)

-----

## Quick Start

Here's a simple example of finding the minima of a 2D Müller-Brown potential.

```python
from ca_abc import CurvatureAdaptiveABC
from ca_abc.potentials import StandardMullerBrown2D

# 1. Initialize the potential
potential = StandardMullerBrown2D()

# 2. Configure and run the ABC simulation
abc = CurvatureAdaptiveABC(potential)
abc.run(max_iterations=100)

# 3. Access the results
# The run prints a live summary, but results are stored on the object.
print("\n--- Simulation Results ---")
print(f"Found {len(abc.minima)} unique minima.")
print("Approximate saddles:", abc.saddles)
```

-----

## Configuration

CA-ABC is highly configurable. Key parameters can be passed during initialization. See the `CurvatureAdaptiveABC` docstring for a full list.

| Parameter                       | Default      | Description                                                                                                                              |
| ------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **General** |              |                                                                                                                                          |
| `curvature_method`              | `"bfgs"`     | Method to estimate the Hessian: `"bfgs"`, `"finite_diff"`, `"lanczos"`.                                                                  |
| `biased_atom_indices`           | `None`       | A list of atom indices to apply biases/perturbations to. If `None`, all atoms are used.                                                  |
| **Perturbation** |              |                                                                                                                                          |
| `perturb_type`                  | `"adaptive"` | `"adaptive"` (deterministic), `"adaptive_stochastic"`, `"stochastic"`, or `"none"`.                                                      |
| `default_perturbation_size`     | `0.05`       | The base magnitude of the perturbation step.                                                                                             |
| `min_perturbation_size`         | `None`       | The minimum allowed magnitude for an adaptive perturbation. Defaults to `default_perturbation_size`.                                     |
| **Biasing (Height)** |              |                                                                                                                                          |
| `bias_height_type`              | `"adaptive"` | `"adaptive"` (scales with curvature) or `"fixed"`.                                                                                       |
| `default_bias_height`           | `1.0`        | The base height of the Gaussian bias potential.                                                                                          |
| `min_bias_height`               | `None`       | The minimum allowed height for an adaptive bias. Defaults to `default_bias_height`.                                                      |
| `max_bias_height`               | `None`       | The maximum allowed height for an adaptive bias. Defaults to `default_bias_height`.                                                      |
| **Biasing (Covariance/Width)** |              |                                                                                                                                          |
| `bias_covariance_type`          | `"adaptive"` | `"adaptive"` (uses inverse Hessian) or `"isotropic"` (spherical).                                                                        |
| `default_bias_covariance`       | `1.0`        | The base variance (width) of the Gaussian bias.                                                                                          |
| `min_bias_covariance`           | `None`       | The minimum allowed variance along any principal axis.                                                                                   |
| `max_bias_covariance`           | `None`       | The maximum allowed variance along any principal axis.                                                                                   |
| **Optimization** |              |                                                                                                                                          |
| `descent_convergence_threshold` | `1e-4`       | The force tolerance (`fmax`) for the geometry optimizer to be considered converged.                                                      |
| `min_descent_steps`             | `5`          | The optimizer will run for at least this many steps, even if the force tolerance is met.                                                 |
| `max_descent_steps`             | `600`        | The maximum number of steps allowed for the optimizer in a single descent.                                                               |

-----

## Documentation & API

Full API documentation is available in the code docstrings.

  - `CurvatureAdaptiveABC`: The main simulation controller.
  - `ABCAnalysis`: Tools for plotting and analyzing simulation results.
  - `potentials`: Contains example potentials and base classes for ASE integration.
      - `CanonicalASEPotential`: A powerful wrapper for ASE `Atoms` objects that handles rotations and translations, working in the 3N-6 canonical degrees of freedom.
  - `optimizers`: Contains the backend optimizers (e.g., `FIREOptimizer`, `ScipyOptimizer`).

-----

## Future Directions

This project is in active development. Key future directions include:

  - **Parallelization**: Implement multi-walker schemes where different runs on the PES share a single, global bias list.
  - **Improved Biasing Metrics**: Incorporate descent history and barrier information to enhance bias height determination, inspired by methods from Cao et al.
  - **Transition Network Analysis**: Implement algorithms like ABC-E (Fan et al.) for automated transition network construction and kMC support.
  - **Machine Learning Integration**: Learn the bias covariance from descent information to better handle anharmonic regions.
  - **Hybrid Optimizers**: Dynamically switch between `FIRE` (robust near saddles) and `BFGS` (fast in harmonic regions) for optimal performance.
  - **Expanded Calculator Support**: Add a dedicated LAMMPS calculator interface via ASE.

-----

## Known Issues

  - The BFGS Hessian approximation can degrade in very high dimensions (\>300D), where the Lanczos method may be preferable.
  - For extremely high-dimensional systems (\>1000D), bias storage can be memory-intensive. It is recommended to use `biased_atom_indices` and frequent data dumping to manage memory.
  - The dynamic EMA scaling for bias covariance often performs best when allowed to exceed its expected `(0, 1]` domain, possibly due to Hessian inaccuracies or landscape anharmonicity.

-----

## Contributing

Contributions are welcome\! Please feel free to open an issue to discuss a bug or feature, or submit a pull request.

-----

## Citation

A preprint describing the algorithm and implementation is in preparation. For now, if you use this code in your research, please cite the GitHub repository:

```
Nirenberg, S., Ding, L., & Do, C. (2025). Curvature-Adaptive Autonomous Basin Climbing. GitHub. [https://github.com/SimonNir/ca_abc](https://github.com/SimonNir/ca_abc)
```

-----

## Acknowledgements

This research was supported in part by an appointment to the Oak Ridge National Laboratory Research Student Internships Program, sponsored by the U.S. Department of Energy and administered by the Oak Ridge Institute for Science and Education.

-----

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.