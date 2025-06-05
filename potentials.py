from abc import ABC, abstractmethod
import numpy as np

###############################
# Base Potential Class (Abstract)
###############################

class PotentialEnergySurface(ABC):
    """Abstract base class for potential energy surfaces."""
    
    @abstractmethod
    def potential(self, position: np.ndarray) -> float: 
        """Compute potential energy at given position."""
        pass
        
    @abstractmethod
    def default_starting_position(self) -> np.ndarray:
        """Return default starting position for this PES."""
        pass
        
    def gradient(self, position) -> np.ndarray:
        """Compute analytic gradient at given position, if available
        Raise NotImplementedError if not implemented.
        
        If your potential has no analytical gradient, simply omit this method 
        from your implementation, and the ABC will perform finite-difference.
        """
        raise NotImplementedError("Analytic gradient not implemented for this PES.")

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
    
    def potential(self, x):
        """Compute double well potential with minima at x=-1 and x=1."""
        return 1/6 * (5 * (x**2 - 1))**2
    
    def gradient(self, x):
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
    
    def potential(self, x):
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
        self.A = np.array([-200, -100, -170, 15])
        self.a = np.array([-1, -1, -6.5, 0.7])
        self.b = np.array([0, 0, 11, 0.6])
        self.c = np.array([-10, -10, -6.5, 0.7])
        self.x0 = np.array([1, 0, -0.5, -1])
        self.y0 = np.array([0, 0.5, 1.5, 1])

    def potential(self, pos):
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

    def gradient(self, position):
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
        if (norm := np.linalg.norm(grad)) > 500.: 
            print(f"Warning: Gradient value of {grad} detected as likely unphysically large in magnitude; shrunk to magnitude 500")
            grad = 500 * grad / norm  
            
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
    This helps bridge your abstract PES to ASE's capabilities.
    """
    def __init__(self, ase_atoms, calculator):
        self.atoms = ase_atoms
        self.atoms.set_calculator(calculator)

    def potential(self, position):
        """Compute potential energy at given position using ASE."""
        # Ensure 'position' is a numpy array of correct shape for ASE
        # For N atoms, it should be (N, 3)
        self.atoms.set_positions(position.reshape(-1, 3))
        return self.atoms.get_potential_energy()

    def gradient(self, position):
        """Compute gradient at given position using ASE."""
        self.atoms.set_positions(position.reshape(-1, 3))
        # ASE returns forces, which are negative gradients
        forces = self.atoms.get_forces()
        return -forces.flatten() # Flatten to match your 'position' input shape


class LennardJonesCluster(ASEPotentialEnergySurface):
    def __init__(self, num_atoms, initial_positions=None):
        # Create an ASE Atoms object for LJ atoms
        # 'X' for generic LJ particles
        symbols = ['X'] * num_atoms
        
        if initial_positions is None:
            # Random initial positions (example)
            initial_positions = np.random.rand(num_atoms, 3) * 5.0 
        
        atoms = Atoms(symbols=symbols, positions=initial_positions, pbc=False)
        
        # Attach the ASE LennardJones calculator
        lj_calculator = LennardJones(epsilon=1.0, sigma=1.0, rc=2.5) # Example LJ parameters
        
        super().__init__(atoms, lj_calculator)
        self.num_atoms = num_atoms

    def default_starting_position(self):
        # Implement specific default starting positions for your LJ clusters
        # e.g., slightly perturbed perfect cluster, or random.
        # This should return a 1D numpy array of positions.
        return self.atoms.get_positions().flatten() # Placeholder

    def known_minima(self):
        # Return a list of known global/local minima for this LJ cluster size
        # Each minimum should be a flattened numpy array of positions
        # e.g., for LJ38, you might list its two famous minima.
        return [] # Placeholder

    def known_saddles(self):
        # Return a list of known saddle points for this LJ cluster size
        return [] # Placeholder    
    

