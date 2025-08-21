# Curvature-Adaptive Autonomous Basin Climbing (CA-ABC)

A Python implementation of the Curvature-Adaptive Autonomous Basin Climbing algorithm for efficient exploration of pathways along rugged potential energy surfaces (PES) without CVs or endpoint information.
This is also the only known Python Autonomous Basin Climbing implementation currently in existence, to my knowledge. Please feel free to use for any chemistry, physics, or materials simulations your heart desires!

## Key Features

- **Curvature-adaptive biasing**: Uses BFGS-derived Hessian information to shape anisotropic bias potentials
- **Soft-mode perturbations**: Guides escapes along low-curvature directions for efficient basin transitions
- **On-the-fly Discovery**: Combines minima discovery and transition state identification in a single workflow
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

If you use **CA-ABC** in your research, we kindly ask that you cite the following work:

Nirenberg, S., Ding, L., & Do, C. (2025). *Curvature-Adaptive Autonomous Basin Climbing: Robust On-The-Fly Pathway Sampling for Rugged Energy Landscapes*. *[Journal Name]*.

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

Contributions are welcome\! Please open an issue or pull request on GitHub. Feel free to refer to the Known Issues and Future Directions section below for details. 

-----

## Known Issues

  * BFGS Hessian approximation degrades in extremely high dimensions (\>300D); Lanczos or Rayleigh methods (to be incorporated in future work) likely become preferable
  * ASE interface may require configuration for some calculators
  * dynamic deltas >> 1 often perform far better than when restricted to the expected (0,1] domain; this might be due to the imperfection of the BFGS hessian or anhamonicity of the landscapes

-----

## Future Directions

  * Include Lanczos Hessian estimation for very high (\>300) dimensional spaces, where the BFGS approximation may be imperfect
  * Implement Fan et al.'s ABC-E algorithm for transition networks, along with kMC support
  * Incorporate a 'deterministic mode', mimicking Kushima et al's original strategy more exactly
  * Improve height metric with descent and past barrier information, possibly by incorporating methods like those of Cao et al. 
  * (Machine) Learn bias covariance from descent information, allowing adaptiveness and flattening in anharmonic regions
  * Incorporate dynamic shifts between BFGS and FIRE for speedups (FIRE is best near saddle points; BFGS is orders of magnitude faster elsewhere, but often tunnels through barriers instead of neatly spilling)
  * Set up LAMMPS calculator (very easy in theory via ASE support)
  * Setup deeper parallelization, with multiple different runs at different positions on the PES sharing a single bias list

-----

## Support

For questions or support, please contact simon\_nirenberg@brown.edu

-----

## Acknowledgements
This research was supported in part by an appointment to the Oak Ridge National Laboratory Research Student Internships
Program, sponsored by the U.S. Department of Energy and administered by the Oak Ridge Institute for Science and Education.
