from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
from ase.visualize import view
import numpy as np

def fibonacci_sphere(samples, radius=2.0):
    points = []
    offset = 2.0 / samples
    increment = np.pi * (3.0 - np.sqrt(5))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y*y)
        phi = i * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([x * radius, y * radius, z * radius])

    return np.array(points)

# Setup cluster
num_atoms = 7
positions = fibonacci_sphere(num_atoms, radius=2.0)
atoms = Atoms('Ar' + str(num_atoms), positions=positions)

atoms.calc = LennardJones(epsilon=1.0, sigma=1.0, rc=3.0)

view(atoms)  # visualize initial

# Setup dimer control and min mode atoms wrapper
with DimerControl(initial_eigenmode_method='displacement',
                  displacement_method='vector', logfile=None) as d_control:
    d_atoms = MinModeAtoms(atoms, d_control)

    # Optionally displace atoms slightly to break symmetry / pick mode
    displacement_vector = [[0.0, 0.0, 0.0] for _ in range(num_atoms)]
    displacement_vector[-1][1] = -0.1  # small displacement on last atom in y
    d_atoms.displace(displacement_vector=displacement_vector)

    # Run dimer relaxation to find saddle point
    with MinModeTranslate(d_atoms, trajectory='dimer_lj.traj', logfile=None) as dim_rlx:
        dim_rlx.run(fmax=0.01, steps=500)

print("Saddle point energy:", atoms.get_potential_energy())
view(atoms)  # visualize final saddle point
