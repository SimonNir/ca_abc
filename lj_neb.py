from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from ase.neb import NEB
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# Initial structure
initial = Atoms('Ar7',
    positions=[
        [0, 0, 0],
        [1.1, 0, 0],
        [0.55, 0.95, 0],
        [-0.55, 0.95, 0],
        [-1.1, 0, 0],
        [-0.55, -0.95, 0],
        [0.55, -0.95, 0],
    ])

# Final structure
final = initial.copy()
final.positions[6] += [0, 0, 1.0]  # move one atom upward

calc = LennardJones(epsilon=1.0, sigma=1.0, rc=3.0)

initial.calc = deepcopy(calc)
final.calc = deepcopy(calc)

# Setup images list
images = [initial.copy()]
for _ in range(5):
    img = initial.copy()
    img.calc = deepcopy(calc)
    images.append(img)
final_copy = final.copy()
final_copy.calc = deepcopy(calc)   # Assign calculator here!
images.append(final_copy)

# Create NEB with climbing image
neb = NEB(images, climb=True)
neb.interpolate()

# Just in case, assign calculator again after interpolation
for img in images:
    if img.calc is None:
        img.calc = deepcopy(calc)

opt = BFGS(neb)
opt.run(fmax=1, steps=200)

# Plot energies
energies = [img.get_potential_energy() for img in images]
plt.plot(range(len(energies)), energies, 'o-')
plt.xlabel('Image')
plt.ylabel('Energy (eV)')
plt.title('LJ7 Cluster Transition Path')
plt.show()

from ase.visualize import view
view(images)
