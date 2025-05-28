from langevin_abc import LangevinABC, plot_results
from potentials import DoubleWellPotential1D, MullerBrownPotential2D, GaussianBias
import numpy as np
import matplotlib.pyplot as plt 

class EnhancedABC(GeneralizedABC): 
    def __init__(self, *args, bias_shift_factor=0., **kwargs): 
        super().__init__(*args, **kwargs)
        self.bias_shift_factor=bias_shift_factor

def main(): 
    np.random.seed(42)
    


if __name__ == "__main__":
    main()


# (normalized) Covariance matrix (# ratio of eigenvalues )
# width?

# Success is transition energy closest and cost 

# remember to use covariance not sigma 