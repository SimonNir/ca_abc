from ca_abc import CurvatureAdaptiveABC
from ca_abc.potentials import *
from ca_abc.optimizers import *
from ca_abc.analysis import ABCAnalysis
import numpy as np
from ase import Atoms

def main(): 
    # np.random.seed(4)

    lj = CanonicalLennardJonesCluster(38)

    # lj = Complex1D()

    abc = CurvatureAdaptiveABC(
        potential=lj,
        curvature_method="bfgs",
        dump_every=30000,

        perturb_type="fixed",
        default_perturbation_size=0.1,
        scale_perturb_by_curvature=False,
        # min_perturbation_size=0.005/1.5,
        # max_perturbation_size=0.005*1.5,
     
        bias_height_type="fixed",
        default_bias_height=10,
        # min_bias_height=.76/1.5,
        # max_bias_height= 1.5*.76,

        bias_covariance_type="fixed",
        # default_bias_covariance=0.003025,
        default_bias_covariance=10,
        # min_bias_covariance= 0.018087/1.5,
        # max_bias_covariance= 1.5*0.018087,
        
        use_ema_adaptive_scaling=True,
        conservative_ema_delta=False, 
        struc_uniqueness_rmsd_threshold=1, 
        
        max_descent_steps=1000,
        descent_convergence_threshold=1e-2, 
        max_acceptable_force_mag=1e99,
    )

    opt = ScipyOptimizer(abc, 'BFGS')
    # opt = FIREOptimizer(abc)

    abc.run(max_iterations=20, stopping_minima_number=100, optimizer=opt, verbose=True, save_summary=True)

    # analyzer = ABCAnalysis("abc_data")
    analyzer = ABCAnalysis(abc)
    # analyzer.plot_summary(save_plots=True, plot_type="neither", filename="lj38_summary_ca.png")
    analyzer.plot_diagnostics(save_plots=True, plot_type="neither", filename="lj38_diagnostics_v3.png")

    # print(abc.energy_calls)
    # print(abc.force_calls)

    from ase import Atoms, io
    mins = [Atoms(['Ar']*38, internal_to_cartesian(pos, 38).reshape(-1,3)) for pos in abc.minima]
    io.write("mins_38.xyz", mins)

    traj = [Atoms(['Ar']*38, internal_to_cartesian(pos, 38).reshape(-1,3)) for pos in abc.trajectory]
    io.write("traj_38.xyz", traj)

    # np.savetxt("e_vs_min_ca.txt", abc.energy_calls)

if __name__ == "__main__":
    main()