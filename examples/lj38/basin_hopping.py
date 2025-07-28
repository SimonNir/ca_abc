import numpy as np
from scipy.optimize import basinhopping, linear_sum_assignment
from ca_abc.potentials import CanonicalLennardJonesCluster
from ase import Atoms
from ase.calculators.lj import LennardJones

# ---------------------------- Parameters ----------------------------
np.random.seed(2)

N = 3 # Number of atoms
min_distance = .9
padding = 0.5
stepsize = 1.
rmsd_tol = 1e-1  # Tight RMSD tolerance for uniqueness

lj = CanonicalLennardJonesCluster(N)


# ---------------------- Run Basin Hopping ---------------------------

minimizer_kwargs = dict(method='L-BFGS-B', jac=lj.gradient, options={'gtol': 1e-5, 'disp': False})

result = basinhopping(
    lj.potential,
    x0=lj.default_starting_position().flatten(),
    minimizer_kwargs=minimizer_kwargs,
    niter=200,
    stepsize=stepsize,
    T=2.0,
    callback=record_minimum
)

# ---------------------- Print All Unique Minima ---------------------

print(f"\nTotal unique minima found: {len(minima)}")
for i, (coords, energy) in enumerate(minima):
    print(f"Minimum {i+1:>2}: Energy = {energy:.6f}")
    print(coords)
    print()

from ase import io

# Save all unique minima as a trajectory
traj = [Atoms('X' * N, positions=coords) for coords, _ in minima]
io.write("basin_hopping_minima.xyz", traj)
