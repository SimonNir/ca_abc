# Curvature-Adaptive Autonomous Basin Climbing (CA-ABC)

A Python implementation of the Curvature-Adaptive Autonomous Basin Climbing algorithm for efficient exploration of rugged potential energy surfaces (PES) without requiring predefined collective variables.

## Key Features

- **Curvature-adaptive biasing**: Uses BFGS-derived Hessian information to shape anisotropic bias potentials
- **Soft-mode perturbations**: Guides escapes along low-curvature directions for efficient basin transitions
- **Integrated optimization**: Combines minima discovery and transition state identification in a single workflow
- **ASE integration**: Works seamlessly with Atomic Simulation Environment for atomistic simulations
- **Adaptive parameter tuning**: Automatically adjusts bias parameters based on local PES features

---

## Installation

```bash
pip install -r requirements.txt 
git clone https://github.com/SimonNir/ca_abc.git
cd ca_abc
pip install -e .
```


**Dependencies:**

  * Python 3.7+
  * NumPy
  * SciPy
  * ASE (Atomic Simulation Environment)
  * Matplotlib (for analysis)

-----

## Basic Usage

```python
from ca_abc import CurvatureAdaptiveABC
from potentials import StandardMullerBrown2D

# Initialize with your potential
potential = StandardMullerBrown2D()
abc = CurvatureAdaptiveABC(potential)

# Run the simulation
abc.run(max_iterations=100)

# Access results
print("Found minima:", abc.minima)
print("Approximate saddles:", abc.saddles)
```

-----

## Configuration Options

Key parameters (see `ca_abc.py` for full list):

**Perturbation:**

  * `perturb_type`: "adaptive" (default) or "random"
  * `default_perturbation_size`: Base step size (default 0.05)
  * `scale_perturb_by_curvature`: Whether to scale steps by curvature (default True)

**Biasing:**

  * `bias_height_type`: "adaptive" (default) or "fixed"
  * `default_bias_height`: Base bias height (default 1.0)
  * `bias_covariance_type`: "adaptive" (default) or "isotropic"

**Optimization:**

  * `descent_convergence_threshold`: Force tolerance (default 1e-4)
  * `max_descent_steps`: Maximum steps per descent (default 600)

-----

## Advanced Features

### Custom Optimizers

Use different optimizers by passing an optimizer instance:

```python
from optimizers import ASEOptimizer
optimizer = ASEOptimizer(abc, optimizer_class='BFGS')
abc.run(optimizer=optimizer)
```

### Canonical Coordinates

For molecular systems, use the canonical coordinate system:

```python
from potentials import CanonicalLennardJonesCluster
potential = CanonicalLennardJonesCluster(num_atoms=38)
abc = CurvatureAdaptiveABC(potential)
```

### Analysis and Visualization

Use the built-in analysis tools:

```python
from analysis import ABCAnalysis
analyzer = ABCAnalysis(abc)
analyzer.plot_summary()
```

-----

## Example Workflows

### Basic Exploration

```python
from ca_abc import CurvatureAdaptiveABC
from potentials import DoubleWell1D

abc = CurvatureAdaptiveABC(DoubleWell1D())
abc.run(max_iterations=50)
abc.summarize()
```

### LJ Cluster Exploration

```python
from potentials import LennardJonesCluster

lj38 = LennardJonesCluster(num_atoms=38)
abc = CurvatureAdaptiveABC(lj38,
                           bias_height_type="adaptive",
                           bias_covariance_type="adaptive")
abc.run(max_iterations=200, stopping_minima_number=5)
```

-----

## Citation

If you use CA-ABC in your research, please cite:

Nirenberg, S., Ding, L., & Do, C. (2023). Curvature-Adaptive Autonomous Basin Climbing: Efficient Exploration of Rugged Energy Landscapes Without Collective Variables. \[Journal Name]

-----

## License

This project is licensed under the MIT License - see the LICENSE file for details.

-----

## Documentation

Full API documentation is available in the code docstrings. Key classes:

  * `CurvatureAdaptiveABC`: Main simulation class
  * `GaussianBias`: Implements anisotropic bias potentials
  * `ABCAnalysis`: Visualization and analysis tools

-----

## Contributing

Contributions are welcome\! Please open an issue or pull request on GitHub.

-----

## Known Issues

  * BFGS Hessian approximation degrades in extremely high dimensions (\>100D); Lanczos or Rayleigh methods likely become preferable 
  * ASE interface may require configuration for some calculators

-----

## Support

For questions or support, please contact simon\_nirenberg@brown.edu