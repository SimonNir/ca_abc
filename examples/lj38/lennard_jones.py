from ca_abc.ca_abc import CurvatureAdaptiveABC
from potentials import *
from src.ca_abc.optimizers import *
from analysis import ABCAnalysis
import numpy as np
from ase import Atoms

def main(): 
    np.random.seed(4)

    lj = CanonicalLennardJonesCluster(13)

    # lj = Complex1D()

    abc = CurvatureAdaptiveABC(
                lj, 
                curvature_method="bfgs", 
                bias_height_type="adaptive",
                         
                max_descent_steps=1000, 
                dump_every=1000,
                descent_convergence_threshold=1e-2, 

                default_bias_height=0.03,
                max_bias_height=0.15, 

                perturb_type="adaptive", 
                bias_covariance_type="adaptive", 
                default_bias_covariance=0.05,
                max_bias_covariance=0.25, 
                use_ema_adaptive_scaling=True,

                default_perturbation_size=0.05,
                scale_perturb_by_curvature=False,
                )

    opt = ScipyOptimizer(abc, 'BFGS')
    # opt = FIREOptimizer(abc)

    abc.run(max_iterations=1000, stopping_minima_number=10, optimizer=opt, verbose=False, save_summary=True)

    # analyzer = ABCAnalysis("abc_data")
    analyzer = ABCAnalysis(abc)
    # analyzer.plot_summary(save_plots=True, plot_type="neither", filename="lj38_summary_ca.png")
    analyzer.plot_diagnostics(save_plots=True, plot_type="neither", filename="lj13_diagnostics_v3.png")

    # print(abc.energy_calls)
    # print(abc.force_calls)

    from ase import Atoms, io
    mins = [Atoms(['X']*13, internal_to_cartesian(pos, 13).reshape(-1,3)) for pos in abc.minima]
    io.write("mins_13_new.xyz", mins)

    # traj = [Atoms(['X']*13, internal_to_cartesian(pos, 13).reshape(-1,3)) for pos in abc.trajectory]
    # io.write("traj_13_v3.xyz", traj)

    # np.savetxt("e_vs_min_ca.txt", abc.energy_calls)

if __name__ == "__main__":
    main()