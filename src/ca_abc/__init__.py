# ca_abc/__init__.py

"""
Curvature-Adaptive Autonomous Basin Climbing (CA-ABC)

A Python package for the efficient exploration of potential energy surfaces.
This library implements the Autonomous Basin Climbing algorithm, enhanced with
on-the-fly, curvature-adaptive biasing and perturbation strategies to discover
minima and transition states without requiring collective variables.
"""

__version__ = "0.1.0"
__author__ = "Simon Nirenberg"
__email__ = "simon_nirenberg@brown.edu"

# --- Public API Imports ---
from .ca_abc import CurvatureAdaptiveABC
from .analysis import ABCAnalysis
from .bias import GaussianBias
from .optimizers import FIREOptimizer, ScipyOptimizer
from .potentials import (
    StandardMullerBrown2D,
    CanonicalASEPotential,
)

__all__ = [
    "CurvatureAdaptiveABC",
    "ABCAnalysis",
    "GaussianBias",
    "FIREOptimizer",
    "ScipyOptimizer",
    "StandardMullerBrown2D",
    "CanonicalASEPotential",
]