from ca_abc import CurvatureAdaptiveABC
from potentials import *
from optimizers import *
from analysis import ABCAnalysis
import numpy as np
from ase import Atoms

def main(): 
    np.random.seed(4)

    lj = CanonicalLennardJonesCluster(13)

    # lj = Complex1D()

    abc = CurvatureAdaptiveABC(
                   lj, 
                   curvature_method="none", 
                   bias_height_type="fixed",
                   bias_covariance_type="fixed", 

                   perturb_type="fixed", 
                
                   max_descent_steps=1000, 
                   dump_every=10000,
                   descent_convergence_threshold=1e-2, 

                   default_bias_height=0.2,
                   max_bias_height=5,
                   min_bias_height=0.05,
                #    curvature_bias_height_scale=1e-2,

                   default_bias_covariance=0.1,

                   default_perturbation_size=0.05,
                   scale_perturb_by_curvature=False, 
                   curvature_perturbation_scale=1, 
                   random_perturb_every=300000, 

                   )
    
    opt = ScipyOptimizer(abc, 'BFGS')
    # opt = SimpleGradientDescent(abc, step_size=1e-1)
    # opt = FIREOptimizer(abc)
    # opt = ASEOptimizer(abc, "BFGS")
    # opt = CanonicalASEOptimizer(abc, "FIRE")

    abc.run(max_iterations=100, optimizer=opt, verbose=True)

    # analyzer = ABCAnalysis("abc_data")
    analyzer = ABCAnalysis(abc)
    analyzer.plot_summary(save_plots=True, plot_type="neither", filename="lj13_summary.png")
    # analyzer.plot_diagnostics(save_plots=True, plot_type="neither", filename="lj13_diagnostics.png")

    from ase import Atoms, io
    mins = [Atoms(['X']*13, internal_to_cartesian(pos, 13).reshape(-1,3)) for pos in abc.minima]
    io.write("mins_13.xyz", mins)

    traj = [Atoms(['X']*13, internal_to_cartesian(pos, 13).reshape(-1,3)) for pos in abc.trajectory]
    io.write("traj_13.xyz", traj)


if __name__ == "__main__":
    main()