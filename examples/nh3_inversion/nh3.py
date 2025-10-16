import os
import numpy as np

# ASE imports for handling atoms and the base calculator class
from ase.io import read
from ase.calculators.calculator import Calculator, all_changes

# PySCF imports for the electronic structure calculation
from pyscf import gto, dft

# CA-ABC imports for the simulation
from ca_abc import CurvatureAdaptiveABC, ABCAnalysis
from ca_abc.potentials import CanonicalASEPotential, align_to_canonical, cartesian_to_internal
from ca_abc.optimizers import FIREOptimizer

# --- 1. Custom ASE Calculator for PySCF ---
# This class teaches ASE how to get energy and forces from PySCF.
class SimplePySCF(Calculator):
    """
    A simplified, self-contained ASE calculator for PySCF.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, xc='PBE', basis='sto-3g', **kwargs):
        """Initializes the calculator with DFT parameters."""
        super().__init__(**kwargs)
        self.xc = xc
        self.basis = basis
        self.mol = None
        self.mf = None

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        """
        This is the core method where ASE asks for a calculation.
        """
        super().calculate(atoms, properties, system_changes)

        # --- Convert ASE Atoms to PySCF Mole object ---
        # PySCF takes atom info as a list of lists: [['N', (x, y, z)], ...]
        pyscf_atom_str = [
            [symbol, pos] for symbol, pos in zip(self.atoms.get_chemical_symbols(), self.atoms.get_positions())
        ]

        mol = gto.Mole()
        mol.atom = pyscf_atom_str
        mol.basis = self.basis
        mol.build()

        # --- Perform the DFT Calculation ---
        # We use Restricted Kohn-Sham (RKS) for this closed-shell molecule.
        mf = dft.RKS(mol)
        mf.xc = self.xc
        
        # Run the self-consistent field calculation and get the energy
        energy = mf.kernel()
        self.results['energy'] = energy

        # --- Calculate Forces (if requested) ---
        if 'forces' in properties:
            # Get the nuclear gradients from PySCF
            grad_method = mf.nuc_grad_method()
            gradients = grad_method.kernel()
            
            # ASE's 'forces' are the negative of the gradients
            self.results['forces'] = -gradients



# --- 3. Set up the ASE Atoms Object with the Custom Calculator ---
print("Setting up ASE Atoms object with our custom PySCF calculator...")

ammonia_atoms = read("nh3.xyz")

# Instantiate our new custom calculator
pyscf_calculator = SimplePySCF(xc='PBE', basis='sto-3g')

# Attach the calculator to the ASE Atoms object
ammonia_atoms.calc = pyscf_calculator

# --- 4. Wrap the ASE system in the CanonicalASEPES ---
print("Wrapping the system in CanonicalASEPES...")
pes = CanonicalASEPotential(ammonia_atoms)

pos = cartesian_to_internal(align_to_canonical(ammonia_atoms.positions.copy()))

# --- 5. Configure and run the CA-ABC simulation ---
print("Configuring the CA-ABC simulation...")
abc_simulation = CurvatureAdaptiveABC(
    starting_position=pos,
    potential=pes,
    biased_atom_indices=None, # Bias fully within reduced canonical space
    curvature_method="none",
    perturb_type="stochastic",
    bias_height_type="fixed",
    bias_covariance_type="fixed",
    default_perturbation_size=0.1,
    default_bias_height=0.1,
    default_bias_covariance=0.05,
    descent_convergence_threshold=0.05,
    max_descent_steps=150,
)

optimizer = FIREOptimizer(abc_simulation, dt=0.05)

# --- 6. Execute the Simulation ---
print(f"\nStarting CA-ABC run for {len(pes.atoms)} atoms ({abc_simulation.dimension} DoF)...")
print("-" * 60)

abc_simulation.run(
    optimizer=optimizer,
    max_iterations=20,
    verbose=True,
    stopping_minima_number=2,
)

analyzer = ABCAnalysis(abc_simulation)
analyzer.plot_summary(save_plots=True, plot_type="neither", filename="nh3_summary.png")
analyzer.plot_diagnostics(save_plots=True, plot_type="neither", filename="nh3_diagnostics.png")

# --- 7. Print a Summary of the Results ---
print("-" * 60)
print("\n --- Simulation Finished --- ")
print(f"Found {len(abc_simulation.minima)} unique minima.")
print(f"Found {len(abc_simulation.saddles)} approximate saddle points.")

from ase.io import write
from ca_abc.potentials import internal_to_cartesian    
    
structures = []
template = abc_simulation.potential.atoms
for i, x in enumerate(abc_simulation.trajectory):
    energy = abc_simulation.unbiased_energies[i]
    atoms = template.copy()
    atoms.set_positions(internal_to_cartesian(x))
    atoms.info['energy'] = energy
    structures.append(atoms)

filename = f"nh3_pbe_sto3g.xyz"
write(filename, structures)
print(f"Saved full trajectory to '{filename}'")