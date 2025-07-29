from ca_abc import CurvatureAdaptiveABC
from ca_abc.potentials import *
from ca_abc.optimizers import *
from ca_abc.analysis import ABCAnalysis
import numpy as np
from ase import Atoms, io

def main(): 
    # g = io.read("/mnt/c/Users/simon/OneDrive - Brown University/Summer 1/ORISE/ca_abc/examples/lj38/wales_global_min.xyz")
    # pos = cartesian_to_internal(align_to_canonical(g.positions.copy()))

    lj = CanonicalLennardJonesCluster(102)

    abc = CurvatureAdaptiveABC(
        # starting_position=pos,
        potential=lj,
        curvature_method="none",
        dump_every=30000,

        perturb_type="fixed",
        default_perturbation_size=0.001,# A
        # default_perturbation_size=0.000001,
        scale_perturb_by_curvature=False,
        # min_perturbation_size=0.005/1.5,
        # max_perturbation_size=0.005*1.5,
     
        bias_height_type="fixed",
        default_bias_height=0.005,# eV
        # default_bias_height=2*0.02,
        # min_bias_height=.76/1.5,
        # max_bias_height= 1.5*.76,

        bias_covariance_type="fixed",
        default_bias_covariance=0.6,# A^2
        # default_bias_covariance=(0.5*0.11)**2,
        # min_bias_covariance= 0.018087/1.5,
        # max_bias_covariance= 1.5*0.018087,
        
        use_ema_adaptive_scaling=True,
        conservative_ema_delta=False, 
        struc_uniqueness_rmsd_threshold=1e-5, 
        
        max_descent_steps=1000,
        descent_convergence_threshold=1e-3, 
        max_acceptable_force_mag=1e99,
    )

    opt = ScipyOptimizer(abc, 'L-BFGS-B')
    opt = ASEOptimizer(abc, 'FIRE')
    # opt = FIREOptimizer(abc)

    abc.run(max_iterations=100, stopping_minima_number=None, 
            ignore_max_steps_on_initial_minimization = True, 
            optimizer=opt, verbose=True, save_summary=True)

    # analyzer = ABCAnalysis("abc_data")
    analyzer = ABCAnalysis(abc)
    # analyzer.plot_summary(save_plots=True, plot_type="neither", filename="lj38_summary_ca.png")
    analyzer.plot_diagnostics(save_plots=True, plot_type="neither", filename="lj102_diagnostics_v3.png")

    # print(abc.energy_calls)
    # print(abc.force_calls)

    mins = [Atoms(['X']*102, internal_to_cartesian(pos, 102).reshape(-1,3)) for pos in abc.minima]
    io.write("mins_102.xyz", mins)

    traj = [Atoms(['X']*102, internal_to_cartesian(pos, 102).reshape(-1,3)) for pos in abc.trajectory]
    io.write("traj_102.xyz", traj)

    # np.savetxt("e_vs_min_ca.txt", abc.energy_calls)

if __name__ == "__main__":
    main()