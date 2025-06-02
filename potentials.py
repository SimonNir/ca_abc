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

    
    def gradient(self, position):
        """Compute analytic gradient at given position, if available
        Raise NotImplementedError if not implemented.
        
        If your potential has no analytical gradient, simply omit this method 
        from your implementation, and the ABC will perform finite-difference.
        """
        raise NotImplementedError("Analytic gradient not implemented for this PES.")
        
    @abstractmethod
    def default_starting_position(self):
        """Return default starting position for this PES."""
        pass
        
    @abstractmethod
    def plot_range(self):
        """Return plotting range for visualization."""
        pass
        
    @abstractmethod
    def known_minima(self):
        """Return known basins (for analysis)."""
        pass

    @abstractmethod
    def known_saddles(self):
        """Return known saddles (for analysis)."""
        pass


###############################
# Concrete Potential Implementations
###############################

class DoubleWell1D(PotentialEnergySurface):
    """1D double well potential."""
    
    def potential(self, x):
        """Compute double well potential with minima at x=-1 and x=1."""
        return 1/6 * (5 * (x**2 - 1))**2
    
    def gradient(self, x):
        return 50/3 * x * (x**2-1)
        
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
    
# class RandomizedMullerBrown2D(PotentialEnergySurface):
#     """Randomized 2D Muller Brown-style potentials"""

#     def potential(self,pos):
#         x, y = pos[0], pos[1]
#         A = np.array([-200, -100, -170, 15])
#         a = np.array([-1, -1, -6.5, 0.7])
#         b = np.array([0, 0, 11, 0.6])
#         c = np.array([-10, -10, -6.5, 0.7])
#         x0 = np.array([1, 0, -0.5, -1])
#         y0 = np.array([0, 0.5, 1.5, 1])

#         V = 0.0
#         for i in range(4):
#             exponent = a[i]*(x - x0[i])**2 + b[i]*(x - x0[i])*(y - y0[i]) + c[i]*(y - y0[i])**2
#             exponent = np.clip(exponent, -100, 100)
#             V += A[i] * np.exp(exponent)
#         return V