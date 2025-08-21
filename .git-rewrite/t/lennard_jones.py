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
                curvature_method="bfgs", 
                bias_height_type="fixed",
                         
                max_descent_steps=1000, 
                dump_every=1000,
                descent_convergence_threshold=1e-2, 

                default_bias_height=0.03,

                perturb_type="fixed", 
                bias_covariance_type="fixed", 
                default_bias_covariance=0.05,
                max_bias_covariance=0.3, 
                min_bias_covariance=0.05, 
                bias_delta=0.1, 

                default_perturbation_size=0.05,
                scale_perturb_by_curvature=False,
                )

    opt = ScipyOptimizer(abc, 'BFGS')
    # opt = SimpleGradientDescent(abc, step_size=1e-1)
    # opt = FIREOptimizer(abc)
    # opt = ASEOptimizer(abc, "FIRE")
    # opt = CanonicalASEOptimizer(abc, "FIRE")

    abc.run(max_iterations=250, optimizer=opt, verbose=False, save_summary=True)

    # analyzer = ABCAnalysis("abc_data")
    analyzer = ABCAnalysis(abc)
    # analyzer.plot_summary(save_plots=True, plot_type="neither", filename="lj38_summary_ca.png")
    analyzer.plot_diagnostics(save_plots=True, plot_type="neither", filename="lj13_diagnostics_v3.png")

    print(abc.energy_calls)
    print(abc.force_calls)

    from ase import Atoms, io
    mins = [Atoms(['X']*13, internal_to_cartesian(pos, 13).reshape(-1,3)) for pos in abc.minima]
    io.write("mins_13_v3.xyz", mins)

    traj = [Atoms(['X']*13, internal_to_cartesian(pos, 13).reshape(-1,3)) for pos in abc.trajectory]
    io.write("traj_13_v3.xyz", traj)

    # np.savetxt("e_vs_min_ca.txt", abc.energy_calls)

if __name__ == "__main__":
    main()