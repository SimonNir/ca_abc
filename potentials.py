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
        if (norm := np.linalg.norm(grad := self._gradient(position))) > self.max_acceptable_force_mag: 
            print(f"Warning: Gradient value of {grad} detected as likely unphysically large in magnitude; shrunk to magnitude {self.max_acceptable_force_mag}")
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
        """Compute the Muller-Brown potential with numerical safeguards."""
        x, y = pos[0], pos[1]

        V = 0.0
        for i in range(4):
            dx = x - self.x0[i]
            dy = y - self.y0[i]
            exponent = self.a[i]*dx**2 + self.b[i]*dx*dy + self.c[i]*dy**2
            exponent = np.clip(exponent, -100, 100)
            V += self.A[i] * np.exp(exponent)
        return V

    def _gradient(self, position):
        x, y = position[0], position[1]

        dVdx = 0.0
        dVdy = 0.0

        for i in range(4):
            dx = x - self.x0[i]
            dy = y - self.y0[i]
            exponent = self.a[i]*dx**2 + self.b[i]*dx*dy + self.c[i]*dy**2
            exp_term = np.exp(np.clip(exponent, -100, 100))
            dVdx += self.A[i] * exp_term * (2*self.a[i]*dx + self.b[i]*dy)
            dVdy += self.A[i] * exp_term * (self.b[i]*dx + 2*self.c[i]*dy)
        
        grad = np.array([dVdx, dVdy])            
        return grad 

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
from potentials import ASEPotentialEnergySurface  # or your custom PES wrapper

class LennardJonesCluster(ASEPotentialEnergySurface):
    def __init__(self, num_atoms, initial_positions=None,
                 sigma=1.0, epsilon=1.0, min_distance=0.9, padding=0.5,
                 barrier_strength=10.0):
        """
        Smarter Lennard-Jones cluster with optional boundary penalty.

        Args:
            num_atoms: Number of atoms
            initial_positions: Starting positions, or None to generate
            sigma: LJ σ parameter
            epsilon: LJ ε parameter
            min_distance: Minimum spacing between atoms (in σ units)
            padding: Box padding around typical cluster size (in σ units)
            barrier_strength: Strength of the soft wall boundary penalty
        """
        self.num_atoms = num_atoms
        self.sigma = sigma
        self.epsilon = epsilon
        self.min_distance = min_distance * sigma
        self.padding = padding * sigma
        self.barrier_strength = barrier_strength

        # Determine bounding box
        self.box_size = self._calculate_box_size()
        self.half_box = self.box_size / 2

        # Generate initial positions
        if initial_positions is None:
            initial_positions = self.default_starting_position()
        initial_positions = np.array(initial_positions).reshape(-1, 3)

        # Create atoms and assign calculator
        atoms = Atoms('X' * num_atoms, positions=initial_positions, pbc=False)
        atoms.calc = LennardJones(sigma=sigma, epsilon=epsilon, rc=3 * sigma, smooth=True)

        super().__init__(atoms, None)

    def _calculate_box_size(self):
        """Estimate a reasonable box size based on density and padding."""
        volume_per_atom = (4 / 3) * np.pi * (self.min_distance / 2)**3
        total_volume = self.num_atoms * volume_per_atom
        linear_size = total_volume**(1 / 3)
        return linear_size + 2 * self.padding

    def default_starting_position(self):
        """Generate valid initial positions inside the bounding box."""
        positions = []
        attempts = 0
        while len(positions) < self.num_atoms:
            trial = np.random.uniform(-self.half_box, self.half_box, 3)
            if all(np.linalg.norm(trial - np.array(p)) >= self.min_distance for p in positions):
                positions.append(trial)
            attempts += 1
            if attempts > 5000:
                raise RuntimeError("Failed to generate non-overlapping initial positions.")
        positions = np.array(positions)
        positions -= positions.mean(axis=0)  # Center cluster
        return positions.flatten()

    def _potential(self, position):
        """Compute potential energy at given position using ASE."""
        # Ensure 'position' is a numpy array of correct shape for ASE
        # For N atoms, it should be (N, 3)
        self.atoms.positions = position.reshape(-1, 3)
        return self.atoms.get_potential_energy() + self._boundary_penalty(position.reshape(-1, 3))

    def _gradient(self, position):
        """Compute gradient at given position using ASE."""
        self.atoms.positions = position.reshape(-1, 3)
        # ASE returns forces, which are negative gradients
        forces = self.atoms.get_forces() - self._boundary_penalty_gradient(position.reshape(-1, 3))
        return -forces.flatten() # Flatten to match your 'position' input shape

    def _boundary_penalty(self, positions):
        """Vectorized soft quartic wall potential to prevent atoms from escaping box."""
        # positions: (N, 3)
        over = np.abs(positions) - self.half_box
        mask = over > 0
        penalty = self.barrier_strength * np.sum(over[mask] ** 4)
        return penalty

    def _boundary_penalty_gradient(self, positions):
        """Vectorized gradient of the soft wall potential."""
        over = np.abs(positions) - self.half_box
        mask = over > 0
        grad = np.zeros_like(positions)
        # Only apply where mask is True
        grad[mask] = 4 * self.barrier_strength * (over[mask] ** 3) * np.sign(positions[mask])
        return grad

    def known_minima(self):
        """Return known configurations for testing small systems."""
        if self.num_atoms == 2:
            return [np.array([0, 0, 0, 0, 0, 1.12 * self.sigma])]
        elif self.num_atoms == 3:
            a = 1.12 * self.sigma
            return [np.array([
                0, 0, 0,
                0, 0.5 * a, 0.866 * a,
                0, -0.5 * a, 0.866 * a
            ])]
        return []

    def known_saddles(self):
        return []