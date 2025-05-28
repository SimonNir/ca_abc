from abc import ABC, abstractmethod
import numpy as np

###############################
# Base Potential Class (Abstract)
###############################

class PotentialEnergySurface(ABC):
    """Abstract base class for potential energy surfaces."""
    
    @abstractmethod
    def potential(self, position):
        """Compute potential energy at given position."""
        pass
        
    @abstractmethod
    def default_starting_position(self):
        """Return default starting position for this PES."""
        pass
        
    @abstractmethod
    def plot_range(self):
        """Return plotting range for visualization."""
        pass
        
    @abstractmethod
    def known_basins(self):
        """Return known basins (for analysis)."""
        pass

###############################
# Concrete Potential Implementations
###############################

class DoubleWellPotential1D(PotentialEnergySurface):
    """1D double well potential."""
    
    def potential(self, x):
        """Compute double well potential with minima at x=-1 and x=1."""
        return 1/6 * (5 * (x**2 - 1))**2
        
    def default_starting_position(self):
        return np.array([-1.0], dtype=float)
        
    def plot_range(self):
        return (-2, 2)
        
    def known_basins(self):
        return [np.array([-1.0], dtype=float), np.array([1.0], dtype=float)]

class MullerBrownPotential2D(PotentialEnergySurface):
    """2D Muller-Brown potential."""
    
    def potential(self, pos):
        """Compute the Muller-Brown potential with numerical safeguards."""
        x, y = pos[0], pos[1]
        A = np.array([-200, -100, -170, 15])
        a = np.array([-1, -1, -6.5, 0.7])
        b = np.array([0, 0, 11, 0.6])
        c = np.array([-10, -10, -6.5, 0.7])
        x0 = np.array([1, 0, -0.5, -1])
        y0 = np.array([0, 0.5, 1.5, 1])
        
        V = 0.0
        for i in range(4):
            exponent = a[i]*(x - x0[i])**2 + b[i]*(x - x0[i])*(y - y0[i]) + c[i]*(y - y0[i])**2
            exponent = np.clip(exponent, -100, 100)
            V += A[i] * np.exp(exponent)
        return V
        
    def default_starting_position(self):
        return np.array([0.0, 0.0], dtype=float)
        
    def plot_range(self):
        return ((-2, 2), (-1, 2))
        
    def known_basins(self):
        return [
            np.array([-0.558, 1.442]),  # Basin A
            np.array([0.623, 0.028]),   # Basin B  
            np.array([-0.050, 0.467])   # Basin C
        ]
    
# class RandomizedMullerBrown2D(PotentialEnergySurface):

