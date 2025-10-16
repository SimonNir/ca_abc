import os
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import BFGS
from ase.calculators.eam import EAM
from ase.vibrations import Vibrations
from copy import deepcopy
import numpy as np

# Load initial and final minima
minima = read('al_surface_minima.xyz@0:2')
initial, final = minima[0], minima[1]

# Attach EAM potential
calc = EAM(potential='Al99.eam.alloy')
for atoms in (initial, final):
    atoms.calc = deepcopy(calc)

mep_file = 'neb_MEP.xyz'
if os.path.exists(mep_file):
    # Load existing NEB path
    print(f"Loading existing NEB path from {mep_file}...")
    images = read(mep_file, index=':')  # load all frames
else:
    # Set up NEB images
    n_images = 7
    images = [initial]
    for i in range(n_images - 2):
        img = initial.copy()
        img.calc = deepcopy(calc)
        images.append(img)
    images.append(final)

    # NEB with climbing image
    neb = NEB(images, climb=True)
    neb.interpolate(method='linear')

    # Run NEB optimization
    opt = BFGS(neb, trajectory="neb_traj.traj", logfile='neb.log', maxstep=0.1)
    opt.run(fmax=0.01)

    # Save NEB path
    write(mep_file, images)

# Extract energies and CI barrier
energies = [atoms.get_potential_energy() for atoms in images]
barrier = max(energies) - energies[0]
print(f"CI-NEB barrier: {barrier:.6f} eV")

# ----------------------------
# Compute Hessian for highest-energy image
# ----------------------------
# Identify highest-energy (climbing) image
ci_index = np.argmax(energies)
ci_image = images[ci_index]

# Reattach calculator if loaded from file
ci_image.calc = deepcopy(calc)

# Only displace the relevant atoms: -1 and -26
moving_atoms = range(-26, 0)

ci_image.calc = deepcopy(calc)

vib = Vibrations(ci_image, indices=moving_atoms, delta=0.01)
vib.run()

# Get vibrational frequencies in eV (imaginary modes will be negative)
freqs = vib.get_frequencies()
vib.clean()

print("Frequencies (eV) for moving atoms:", freqs)
n_imag = np.sum(freqs < 0)
print(f"Number of imaginary frequencies: {n_imag}")

if n_imag == 1:
    print("This is a proper first-order saddle point (transition state).")
else:
    print("This is not a proper transition state.")
