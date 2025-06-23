from smart_abc import SmartABC
from potentials import LennardJonesCluster
from optimizers import ScipyOptimizer
from analysis import ABCAnalysis
import numpy as np

def main(): 
    np.random.seed(21)

    lj = LennardJonesCluster(num_atoms=2)

    abc = SmartABC(
                   lj, 
                   dump_every=0, 
                   curvature_method="full_hessians", 
                   bias_type="constant", 
                   perturb_type="random", 
                   max_descent_steps=150, 
                   descent_convergence_threshold=0.001, 
                   default_bias_height=0.05,
                   default_bias_covariance=0.1,
                   default_perturbation_size=0.1,
                   )
    
    opt = ScipyOptimizer(abc, 'BFGS')
    abc.run(max_iterations=10, optimizer=opt, verbose=False)

    # analyzer = ABCAnalysis("abc_data")
    analyzer = ABCAnalysis(abc)
    analyzer.plot_summary(save_plots=True, filename="lj38.png")

if __name__ == "__main__":
    main()