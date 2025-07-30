from abc import ABC, abstractmethod
import numpy as np

###############################
# Base Potential Class (Abstract)
###############################

class PotentialEnergySurface(ABC):
    """Abstract base class for potential energy surfaces."""

    def __init__(self): 
        self.max_acceptable_force_mag = np.inf # will be updated by ABC later
        self.energy_calls = 0 
        self.force_calls = 0
    
    @abstractmethod
    def _potential(self, position: np.ndarray) -> float: 
        """Compute potential energy at given position."""
        pass

    def potential(self, position: np.ndarray) -> float: 
        """
        Wrapper for potential calls. 
        
        DO NOT EDIT: Users should implement _potential()
        """
        self.energy_calls += 1
        return self._potential(position)
        
    @abstractmethod
    def default_starting_position(self) -> np.ndarray:
        """Return default starting position for this PES."""
        pass
    
    def _gradient(self, position) -> np.ndarray:
        """Compute analytic gradient at given position, if available
        Raise NotImplementedError if not implemented.
        
        If your potential has no analytical gradient, simply omit this method 
        from your implementation, and the ABC will perform finite-difference.
        """
        raise NotImplementedError("Analytic gradient not implemented for this PES.")
    
    def gradient(self, position) -> np.ndarray:
        """
        Wrapper for _gradient with built-in force magnitude limiting
        DO NOT EDIT: Users should implement _gradient() for analytical gradient calculation
        """
        try:
            grad = self._gradient(position)
        except Exception as e:
            print(f"Exception in gradient calculation: {e}")
            grad = np.zeros_like(position)
        
        # Add NaN check
        if np.any(np.isnan(grad)):
            print("NaN detected in gradient! Resetting to zeros.")
            grad = np.zeros_like(position)
        
        norm = np.linalg.norm(grad)
        if norm > self.max_acceptable_force_mag: 
            print(f"Warning: Gradient magnitude {norm:.2e} exceeds limit, clipping")
            grad = self.max_acceptable_force_mag * grad / norm
        
        self.force_calls += 1
        return grad     

    def plot_range(self) -> tuple:
        """Return plotting range for visualization."""
        return None
        
    def known_minima(self) -> list[np.ndarray]:
        """Return known basins (for analysis)."""
        return None 

    def known_saddles(self) -> list[np.ndarray]:
        """Return known saddles (for analysis)."""
        return None


###############################
# Concrete Potential Implementations
###############################

class DoubleWell1D(PotentialEnergySurface):
    """1D double well potential."""
    
    def _potential(self, x):
        """Compute double well potential with minima at x=-1 and x=1."""
        return 1/6 * (5 * (x**2 - 1))**2
    
    def _gradient(self, x):
        return np.array([50/3 * x * (x**2-1)])
        
    def default_starting_position(self):
        return np.array([-1.0], dtype=float)
        
    def plot_range(self):
        return (-2, 2)
        
    def known_minima(self):
        return [np.array([-1.0], dtype=float), np.array([1.0], dtype=float)]
    
    def known_saddles(self):
        return [np.array([0.0], dtype=float)]

class Complex1D(PotentialEnergySurface):
    
    def _potential(self, x):
        a=[6.5, 4.2, -7.3, -125]
        b=[2.5, 4.3, 1.5, 0.036]
        c=[9.7, 1.9, -2.5, 12]
        V = x**2
        for i in range(4):
            exponent = -b[i]*(x-c[i])**2
            exponent = np.clip(exponent, -100, 100)
            V += a[i]*np.exp(exponent)
        return V
     
    def default_starting_position(self):
        return np.array([0.0], dtype=float)
        
    def plot_range(self):
        return (-3.5, 11.6)
        
    def known_minima(self):
        return [
                np.array([-2.27151]), 
                np.array([0.41295]), 
                np.array([2.71638]), 
                np.array([8.69999]), 
                np.array([10.35518]) 
                ]
    
    def known_saddles(self):
        return [
                np.array([-1.2645]),
                np.array([1.94219]), 
                np.array([4.55508]),
                np.array([9.7913])
                ]


import numpy as np

import numpy as np

class StandardMullerBrown2D(PotentialEnergySurface):
    """2D Muller-Brown potential."""

    def __init__(self):
        super().__init__()
        self.A = np.array([-200, -100, -170, 15])
        self.a = np.array([-1, -1, -6.5, 0.7])
        self.b = np.array([0, 0, 11, 0.6])
        self.c = np.array([-10, -10, -6.5, 0.7])
        self.x0 = np.array([1, 0, -0.5, -1])
        self.y0 = np.array([0, 0.5, 1.5, 1])

    def _potential(self, pos):
        x, y = pos[0], pos[1]

        dx = x - self.x0  # shape (4,)
        dy = y - self.y0  # shape (4,)

        exponent = self.a * dx**2 + self.b * dx * dy + self.c * dy**2
        exponent = np.clip(exponent, -100, 100)

        V = np.sum(self.A * np.exp(exponent))
        return V

    def _gradient(self, position):
        x, y = position[0], position[1]

        dx = x - self.x0  # (4,)
        dy = y - self.y0  # (4,)

        exponent = self.a * dx**2 + self.b * dx * dy + self.c * dy**2
        exponent = np.clip(exponent, -100, 100)
        exp_term = np.exp(exponent)  # (4,)

        dVdx = np.sum(self.A * exp_term * (2*self.a*dx + self.b*dy))
        dVdy = np.sum(self.A * exp_term * (self.b*dx + 2*self.c*dy))

        return np.array([dVdx, dVdy])

    def default_starting_position(self):
        return np.array([0.0, 0.0], dtype=float)

    def plot_range(self):
        return ((-2, 2), (-1, 2))

    def known_minima(self):
        return [
            np.array([-0.5582236346, 1.441725842]),  # Basin A
            np.array([0.6234994049, 0.02803775853]), # Basin B  
            np.array([-0.050010823, 0.4666941049])   # Basin C
        ]

    def known_saddles(self):
        return [
            np.array([0.212486582, 0.2929883251]),   # Transition A<-->B
            np.array([-0.8220015587, 0.6243128028])  # Transition B<-->C
        ]

# --- Now, a concrete implementation using ASE ---
from ase import Atoms
from ase.calculators.lj import LennardJones

class ASEPotentialEnergySurface(PotentialEnergySurface):
    """
    A base class for PES implementations that use ASE calculators.
    """
    def __init__(self, ase_atoms, calculator):
        super().__init__()
        self.atoms = ase_atoms
        if calculator is not None: 
            self.atoms.calc = calculator

    def _potential(self, position):
        """Compute potential energy at given position using ASE."""
        # Ensure 'position' is a numpy array of correct shape for ASE
        # For N atoms, it should be (N, 3)
        self.atoms.positions = position.reshape(-1, 3)
        return self.atoms.get_potential_energy()

    def _gradient(self, position):
        """Compute gradient at given position using ASE."""
        self.atoms.positions = position.reshape(-1, 3)
        # ASE returns forces, which are negative gradients
        forces = self.atoms.get_forces()
        return -forces.flatten() # Flatten to match your 'position' input shape

from ase import Atoms
from ase.calculators.lj import LennardJones
import numpy as np

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones

# DEPRECATED: USE CANONICAL
# class LennardJonesCluster(ASEPotentialEnergySurface):
#     def __init__(self, num_atoms, initial_positions=None,
#                  sigma=1.0, epsilon=1.0, min_distance=0.9, padding=0.5,
#                  barrier_strength=10.0):
#         """
#         Smarter Lennard-Jones cluster with optional boundary penalty.

#         Args:
#             num_atoms: Number of atoms
#             initial_positions: Starting positions, or None to generate
#             sigma: LJ σ parameter
#             epsilon: LJ ε parameter
#             min_distance: Minimum spacing between atoms (in σ units)
#             padding: Box padding around typical cluster size (in σ units)
#             barrier_strength: Strength of the soft wall boundary penalty
#         """
#         self.num_atoms = num_atoms
#         self.sigma = sigma
#         self.epsilon = epsilon
#         self.min_distance = min_distance * sigma
#         self.padding = padding * sigma
#         self.barrier_strength = barrier_strength

#         # Determine bounding box
#         self.box_size = self._calculate_box_size()
#         self.half_box = self.box_size / 2

#         # Generate initial positions
#         if initial_positions is None:
#             initial_positions = self.default_starting_position()
#         initial_positions = np.array(initial_positions).reshape(-1, 3)

#         # Create atoms and assign calculator
#         atoms = Atoms('X' * num_atoms, positions=initial_positions, pbc=False)
#         atoms.calc = LennardJones(sigma=sigma, epsilon=epsilon, rc=300*sigma, smooth=False)

#         super().__init__(atoms, None)

#     def _calculate_box_size(self):
#         """Estimate a reasonable box size based on density and padding."""
#         volume_per_atom = (4 / 3) * np.pi * (self.min_distance / 2)**3
#         total_volume = self.num_atoms * volume_per_atom
#         linear_size = total_volume**(1 / 3)
#         return linear_size + 2 * self.padding

#     def default_starting_position(self):
#         """Generate valid initial positions inside the bounding box."""
#         positions = []
#         attempts = 0
#         positions = uniform_sphere_points(self.num_atoms)
#         positions -= positions.mean(axis=0)  # Center cluster
#         return positions.flatten()

#     def _potential(self, position):
#         """Compute potential energy at given position using ASE."""
#         # Ensure 'position' is a numpy array of correct shape for ASE
#         # For N atoms, it should be (N, 3)
#         self.atoms.positions = position.reshape(-1, 3)
#         return self.atoms.get_potential_energy() + self._boundary_penalty(position.reshape(-1, 3))

#     def _gradient(self, position):
#         """Compute gradient at given position using ASE."""
#         self.atoms.positions = position.reshape(-1, 3)
#         # ASE returns forces, which are negative gradients
#         forces = self.atoms.get_forces() + self._boundary_penalty_gradient(position.reshape(-1, 3))
#         return -forces.flatten() # Flatten to match your 'position' input shape

#     def _boundary_penalty(self, positions):
#         """Vectorized soft quartic wall potential to prevent atoms from escaping box."""
#         # positions: (N, 3)
#         over = np.abs(positions) - self.half_box
#         mask = over > 0
#         penalty = self.barrier_strength * np.sum(over[mask] ** 4)
#         return penalty

#     def _boundary_penalty_gradient(self, positions):
#         """Vectorized gradient of the soft wall potential."""
#         over = np.abs(positions) - self.half_box
#         mask = over > 0
#         grad = np.zeros_like(positions)
#         # Only apply where mask is True
#         grad[mask] = 4 * self.barrier_strength * (over[mask] ** 3) * np.sign(positions[mask])
#         return grad

#     def known_minima(self):
#         """Return known configurations for testing small systems."""
#         if self.num_atoms == 2:
#             return [np.array([0, 0, 0, 0, 0, 1.12 * self.sigma])]
#         elif self.num_atoms == 3:
#             a = 1.12 * self.sigma
#             return [np.array([
#                 0, 0, 0,
#                 0, 0.5 * a, 0.866 * a,
#                 0, -0.5 * a, 0.866 * a
#             ])]
#         return []

#     def known_saddles(self):
#         return []
    

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones

# === Softplus and Derivatives ===
import numpy as np
from scipy.special import expit  # Stable sigmoid

def softplus(x, k=10):
    """Numerically stable softplus."""
    xk = k * x
    return np.where(
        xk > 50,
        x,
        (np.log1p(np.exp(-np.abs(xk))) + np.maximum(xk, 0)) / k
    )

def d_softplus_dx(x, k=10):
    """Derivative of softplus, i.e., sigmoid."""
    return expit(k * x)

def inverse_softplus(y, k=10):
    """Numerically stable inverse of softplus."""
    y = np.clip(y, 1e-12, None)  # Prevent log of zero/negative
    ky = k * y
    # For very small ky, expm1(ky) ≈ ky, and log(ky) is a good approximation
    small = ky < 1e-5
    log_term = np.where(
        small,
        np.log(ky),  # log(ky) is approx log(expm1(ky)) for small ky
        np.log(np.expm1(ky))
    )
    return np.where(ky > 50, y, log_term / k)


# === Coordinate Transforms ===
def internal_to_cartesian(x_internal, N=None, k=10):
    
    if N is None:
        if len(x_internal) == 1:
            N = 2
        elif len(x_internal) == 3:
            N = 3
        else:
            # Each additional atom beyond 3 contributes 3 internal coordinates
            N = 3 + (len(x_internal) - 3) // 3
    
    pos = np.zeros((N, 3))

    if N == 2:
        # Atom 0 at origin, atom 1 at (softplus(x), 0, 0)
        pos[1, 0] = softplus(x_internal[0], k)
        return pos

    if N == 3:
        pos[1, 0] = softplus(x_internal[0], k)
        pos[2, 0] = x_internal[1]
        pos[2, 1] = softplus(x_internal[2], k)
        return pos

    # General case
    pos[1, 0] = softplus(x_internal[0], k)
    pos[2, 0] = x_internal[1]
    pos[2, 1] = softplus(x_internal[2], k)

    pos[3:] = x_internal[3:].reshape(-1, 3)

    if np.any(np.isnan(pos)):
        print("NaN in cartesian positions! Resetting to generic spherical position.") 
        radius = 1.1 * N ** (1/3)
        points = uniform_sphere_points(N)
        points *= radius
        aligned = align_to_canonical(points)
        return aligned

    return pos

def cartesian_to_internal(pos, k=10):
    N = len(pos)

    if N == 2:
        return np.array([inverse_softplus(pos[1, 0], k)])

    if N == 3:
        return np.array([
            inverse_softplus(pos[1, 0], k),
            pos[2, 0],
            inverse_softplus(pos[2, 1], k)
        ])

    # General case
    x_internal = np.empty(3 * N - 6)
    x_internal[0] = inverse_softplus(pos[1, 0], k)
    x_internal[1] = pos[2, 0]
    x_internal[2] = inverse_softplus(pos[2, 1], k)

    # Flatten the rest
    rest = pos[3:]
    x_internal[3:] = np.column_stack([
        rest[:, 0],
        rest[:, 1],
        rest[:, 2], 
    ]).reshape(-1)

    return x_internal

# === Uniform Sphere Sampling and Canonical Alignment ===
def uniform_sphere_points(n):
    indices = np.arange(0, n) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.vstack((x, y, z)).T


def align_to_canonical(points):
    """
    Perform canonical alignment of points in 3D space with all coordinates >= 0.
    
    Args:
        points: Nx3 numpy array of 3D points
        
    Returns:
        aligned_points: Canonically aligned points with all coordinates >= 0
        rotation_matrix: The combined rotation matrix used
    """
    if len(points) < 1:
        return points, np.eye(3)
    
    # Make a copy to avoid modifying original
    points = np.array(points, dtype=float)
    aligned_points = points.copy()
    
    # Initialize combined rotation matrix
    rotation_matrix = np.eye(3)
    
    # Step 1: Translate so first point is at origin
    translation = aligned_points[0].copy()
    aligned_points -= translation
    
    if len(points) >= 2:
        # Step 2: Rotate so second point is along positive x-axis
        v1 = aligned_points[1]
        if not np.allclose(v1, 0):
            # Normalize
            v1_norm = v1 / np.linalg.norm(v1)
            
            # Find rotation to align v1 with x-axis
            x_axis = np.array([1, 0, 0])
            cross = np.cross(v1_norm, x_axis)
            
            if np.linalg.norm(cross) > 1e-10:  # If not already aligned
                cross = cross / np.linalg.norm(cross)
                dot = np.dot(v1_norm, x_axis)
                angle = np.arccos(np.clip(dot, -1.0, 1.0))
                
                # Rodrigues' rotation formula
                K = np.array([[0, -cross[2], cross[1]],
                            [cross[2], 0, -cross[0]],
                            [-cross[1], cross[0], 0]])
                R1 = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            else:
                # Already aligned or anti-aligned
                if dot < 0:  # Anti-aligned
                    R1 = -np.eye(3)
                    R1[0,0] = 1  # Flip y and z but keep x
                else:  # Already aligned
                    R1 = np.eye(3)
            
            aligned_points = (R1 @ aligned_points.T).T
            rotation_matrix = R1 @ rotation_matrix
    
    if len(points) >= 3:
        # Step 3: Rotate around x-axis so third point has z=0 and y>0
        v2 = aligned_points[2]
        if not np.allclose(v2[1:], 0):  # If not already on x-axis
            # We want to rotate around x-axis to eliminate z component
            yz_norm = np.linalg.norm(v2[1:])
            if yz_norm > 1e-10:
                # Angle needed to rotate point to positive y-axis
                angle = -np.arctan2(v2[2], v2[1])
                
                # Rotation matrix around x-axis
                R2 = np.array([[1, 0, 0],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle), np.cos(angle)]])
                
                aligned_points = (R2 @ aligned_points.T).T
                rotation_matrix = R2 @ rotation_matrix
    
    if len(points) >= 4:
        # Step 4: If fourth point has negative z, flip around xy-plane
        v3 = aligned_points[3]
        if v3[2] < 0:
            # Flip z-coordinates (rotation of 180 degrees around z-axis would also work)
            R3 = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, -1]])
            
            aligned_points = (R3 @ aligned_points.T).T
            rotation_matrix = R3 @ rotation_matrix
    
    return aligned_points

class CanonicalASEPES(PotentialEnergySurface):
    def __init__(self, atoms, k_soft=10):
        super().__init__()
        self.atoms = atoms.copy()
        self.atoms.calc = atoms.calc
        self.N = len(atoms)
        self.k_soft = k_soft

    def _potential(self, x_internal):
        pos = internal_to_cartesian(x_internal, self.N, self.k_soft)
        self.atoms.positions = pos
        return self.atoms.get_potential_energy()

    def _gradient(self, x_internal):
        """
        Compute gradient by properly projecting Cartesian forces into internal coordinates.
        This uses the chain rule: dE/dq = (dE/dr) * (dr/dq)
        where q are internal coordinates and r are Cartesian coordinates.
        """
        pos = internal_to_cartesian(x_internal, self.N, self.k_soft)
        self.atoms.positions = pos
        forces = self.atoms.get_forces()  # This is -dE/dr
        
        # Compute Jacobian matrix dr/dq using finite differences
        jacobian = self._compute_jacobian(x_internal)
        
        # Flatten forces to 1D array
        forces_flat = forces.flatten()
        
        # Apply chain rule: dE/dq = -forces^T * jacobian
        # The negative sign is because forces = -dE/dr
        gradient = -np.dot(forces_flat, jacobian)
        
        return gradient

    def _compute_jacobian(self, x_internal, eps=1e-6):
        N = self.N
        k_soft = self.k_soft
        n_internal = len(x_internal)
        n_cartesian = 3 * N
        jacobian = np.zeros((n_cartesian, n_internal))

        x_perturbed = x_internal.copy()

        for i in range(n_internal):
            orig_val = x_perturbed[i]

            x_perturbed[i] = orig_val + eps
            pos_plus = internal_to_cartesian(x_perturbed, N, k_soft).flatten()

            x_perturbed[i] = orig_val - eps
            pos_minus = internal_to_cartesian(x_perturbed, N, k_soft).flatten()

            jacobian[:, i] = (pos_plus - pos_minus) / (2 * eps)

            x_perturbed[i] = orig_val

        return jacobian

    def _gradient_analytical(self, x_internal):
        """
        DEPRECATED; CANNOT BE TRUSTED
        Alternative analytical gradient implementation for specific cases.
        This is more efficient but requires careful derivation of the transformations.
        """
        pos = internal_to_cartesian(x_internal, self.N, self.k_soft)
        self.atoms.positions = pos
        forces = self.atoms.get_forces()
        grad_full = -forces  # Convert forces to gradients
        grad = np.zeros_like(x_internal)

        if self.N == 2:
            # For 2 atoms: only 1 internal coordinate (distance)
            grad[0] = grad_full[1, 0] * d_softplus_dx(x_internal[0], self.k_soft)
            return grad

        if self.N == 3:
            # For 3 atoms: 3 internal coordinates
            grad[0] = grad_full[1, 0] * d_softplus_dx(x_internal[0], self.k_soft)
            grad[1] = grad_full[2, 0]
            grad[2] = grad_full[2, 1] * d_softplus_dx(x_internal[2], self.k_soft)
            return grad

        # General case for N > 3
        # First three internal coordinates
        grad[0] = grad_full[1, 0] * d_softplus_dx(x_internal[0], self.k_soft)
        grad[1] = grad_full[2, 0]
        grad[2] = grad_full[2, 1] * d_softplus_dx(x_internal[2], self.k_soft)

        # Remaining coordinates - need proper transformation
        if len(x_internal) > 3:
            # For atoms 4 and beyond, we need to account for the full transformation
            # This is where the original code was insufficient
            for i in range(3, self.N):
                idx_start = 3 + (i - 3) * 3
                idx_end = idx_start + 3
                
                if idx_end <= len(x_internal):
                    # Extract the internal coordinates for this atom
                    atom_coords = x_internal[idx_start:idx_end]
                    
                    # Transform gradients properly considering the coordinate system
                    # This requires the full Jacobian transformation
                    grad[idx_start] = grad_full[i, 0]  # x-component
                    grad[idx_start + 1] = grad_full[i, 1]  # y-component
                    grad[idx_start + 2] = grad_full[i, 2] * d_softplus_dx(atom_coords[2], self.k_soft)

        return grad

    def default_starting_position(self) -> np.ndarray:
        """Return default starting position in internal coordinates."""
        # This should be implemented based on your internal coordinate system
        # For now, returning a placeholder
        n_internal = 3 * self.N - 6
        return np.zeros(n_internal)

# === Lennard-Jones Cluster in Canonical Frame ===
class CanonicalLennardJonesCluster(CanonicalASEPES):
    def __init__(self, num_atoms, sigma=1.0, epsilon=1.0, rc=6.5, k_soft=10):
        self.num_atoms = num_atoms
        self.sigma = sigma
        self.epsilon = epsilon
        self.k_soft = k_soft

        atoms = Atoms('X' * num_atoms, pbc=False)
        atoms.calc = LennardJones(sigma=sigma, epsilon=epsilon, rc=rc, smooth=True)

        super().__init__(atoms, k_soft=k_soft)

    def default_starting_position(self):
        N = self.num_atoms
        radius = 1.1 * self.sigma * N ** (1/3)
        points = uniform_sphere_points(N)
        points *= radius
        aligned = align_to_canonical(points)
        return cartesian_to_internal(aligned, k=self.k_soft)
