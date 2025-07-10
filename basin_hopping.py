import numpy as np
from scipy.optimize import basinhopping, linear_sum_assignment
from ase import Atoms
from ase.calculators.lj import LennardJones

# ---------------------------- Parameters ----------------------------
np.random.seed(2)

N = 3 # Number of atoms
min_distance = .9
padding = 0.5
stepsize = 1.
rmsd_tol = 1e-1  # Tight RMSD tolerance for uniqueness


def calculate_box_size():
        """Calculate box size based on particle count and LJ parameters"""
        # Volume scales with number of particles
        volume_per_atom = (4/3) * np.pi * (min_distance/2)**3
        total_volume = N * volume_per_atom
        
        # Cubic root to get linear dimension
        linear_size = total_volume ** (1/3)
        
        # Add padding and convert to box length
        return linear_size + 2*padding

box_size = calculate_box_size()

# ------------------- Rigid-Body Motion Removal ----------------------

def remove_translation(positions):
    return positions - np.mean(positions, axis=0)

def kabsch(P, Q):
    """Kabsch algorithm to find optimal rotation matrix U minimizing RMSD(P, Q)."""
    C = np.dot(P.T, Q)
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    D = np.eye(3)
    D[2, 2] = d
    U = np.dot(np.dot(V, D), Wt)
    return U

def perm_invariant_rmsd(P, Q):
    """
    Calculate RMSD between P and Q with optimal permutation and rotation.
    P, Q: (N,3) arrays.
    """
    N = len(P)
    # Compute squared distance matrix between atoms in P and Q
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist_matrix[i, j] = np.sum((P[i] - Q[j])**2)
    
    # Hungarian algorithm for optimal atom matching
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    # Reorder Q according to assignment
    Q_matched = Q[col_ind]
    
    # Center P and Q_matched
    P_cent = P - P.mean(axis=0)
    Q_cent = Q_matched - Q_matched.mean(axis=0)
    
    # Compute optimal rotation with Kabsch
    U = kabsch(P_cent, Q_cent)
    Q_rot = np.dot(Q_cent, U)
    
    # Compute RMSD
    diff = P_cent - Q_rot
    rmsd_val = np.sqrt(np.sum(diff**2) / N)
    return rmsd_val

def remove_rigid_motion(x):
    coords = x.reshape((N, 3))
    coords = remove_translation(coords)
    return coords.flatten()

# ---------------------------- Setup ---------------------------------

initial_positions = np.random.rand(N, 3) * box_size
cluster = Atoms('Ar' * N, positions=initial_positions)
cluster.calc = LennardJones()

# ---------------------------- Energy and Gradient -------------------

def lj_energy(x):
    x = remove_rigid_motion(x)
    cluster.set_positions(x.reshape((N, 3)))
    return cluster.get_potential_energy()

def lj_gradient(x):
    x = remove_rigid_motion(x)
    cluster.set_positions(x.reshape((N, 3)))
    return (-cluster.get_forces()).flatten()

# ---------------------- Minima Tracking -----------------------------

minima = []

def is_new_minimum(coords):
    for prev_coords, _ in minima:
        if perm_invariant_rmsd(coords, prev_coords) < rmsd_tol:
            return False
    return True

def record_minimum(x, f, accept):
    coords = remove_rigid_motion(x).reshape((N, 3))
    # print coords norm for debug
    # print("Candidate coords norm:", np.linalg.norm(coords))
    
    # Check all stored minima:
    for prev_coords, _ in minima:
        rmsd_val = perm_invariant_rmsd(coords, prev_coords)
        # print(f"RMSD to prev minimum: {rmsd_val:.5f}")
        if rmsd_val < rmsd_tol:
            return  # Duplicate found, skip storing
    
    minima.append((coords.copy(), f))
    print(f"Found new minimum! Energy = {f:.6f} (total: {len(minima)})")


# ---------------------- Run Basin Hopping ---------------------------

minimizer_kwargs = dict(method='L-BFGS-B', jac=lj_gradient, options={'gtol': 1e-5, 'disp': False})

result = basinhopping(
    lj_energy,
    x0=remove_rigid_motion(initial_positions.flatten()),
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
