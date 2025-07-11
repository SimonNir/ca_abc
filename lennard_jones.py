from ca_abc import CurvatureAdaptiveABC
from potentials import *
from optimizers import ScipyOptimizer, ASEOptimizer
from analysis import ABCAnalysis
import numpy as np
from ase import Atoms

def main(): 
    np.random.seed(4)

    lj = LennardJonesCluster(num_atoms=5, barrier_strength=10)

    # lj = DoubleWell1D()

    abc = CurvatureAdaptiveABC(
                   lj, 
                   curvature_method="None", 
                   bias_height_type="fixed",
                   bias_covariance_type="fixed", 
                   perturb_type="fixed", 
                   max_descent_steps=2000, 
                   dump_every=10000,
                   descent_convergence_threshold=1e-5, 
                   default_bias_height=.5,
                   default_bias_covariance=1.5,
                   default_perturbation_size=0.01,
                   remove_rotation_translation=False,
                   )
    
    opt = ScipyOptimizer(abc, 'BFGS')
    abc.run(max_iterations=300, optimizer=opt, verbose=True)

    # analyzer = ABCAnalysis("abc_data")
    analyzer = ABCAnalysis(abc)
    analyzer.plot_summary(save_plots=False, plot_type="neither", filename="lj5.png")

    from ase import Atoms, io
    mins = [Atoms(['X']*5, pos.reshape(-1,3)) for pos in abc.minima]
    io.write("mins_5.xyz", mins)

    traj = [Atoms(['X']*5, pos.reshape(-1,3)) for pos in abc.trajectory]
    io.write("traj_5.xyz", traj)


if __name__ == "__main__":
    main()