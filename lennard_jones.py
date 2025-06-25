from ca_abc import CurvatureAdaptiveABC
from potentials import LennardJonesCluster, ASEPotentialEnergySurface
from optimizers import ScipyOptimizer
from analysis import ABCAnalysis
import numpy as np
from ase import Atoms

def main(): 
    np.random.seed(21)

    lj = LennardJonesCluster(num_atoms=12)

    abc = CurvatureAdaptiveABC(
                   lj, 
                   curvature_method="None", 
                   bias_type="constant", 
                   perturb_type="random", 
                   max_descent_steps=10000000, 
                   descent_convergence_threshold=1e-5, 
                   default_bias_height=1,
                   default_bias_covariance=1,
                   default_perturbation_size=1e-4,
                   max_acceptable_force_mag=1000000, 
                   )
    
    opt = ScipyOptimizer(abc, 'BFGS')
    abc.run(max_iterations=100, optimizer=opt, verbose=False)

    # analyzer = ABCAnalysis("abc_data")
    analyzer = ABCAnalysis(abc)
    analyzer.plot_summary(save_plots=False, filename="lj38.png")

if __name__ == "__main__":
    main()