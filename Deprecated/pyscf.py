import numpy as np
from ase.calculators.calculator import Calculator, all_changes 

from deprecated.pyscf import dft, grad
from deprecated.pyscf import gto as gto_loc 


convert_energy    =  27.2114   # ev/Ha
convert_forces    =  27.2114 / 0.529177 # (Ha/Bohr) * (eV/Ha) * 1/(A/Bohr) = eV/A
convert_positions =  0.529177 # A/Bohr

class PySCF(Calculator):
    """
    A PySCF calculator built to interface with ASE for calculations involving 
    small molecules.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Get positions and atomic numbers from the Atoms object
        positions = self.atoms.positions
        atomic_numbers = self.atoms.numbers
        
        # Calculate energy and forces here using your custom method
        energy = self.compute_energy(positions, atomic_numbers)
        forces = self.compute_forces(positions, atomic_numbers)
        
        # Set results to be used by ASE
        self.results = {'energy': energy, 'forces': forces}

    def compute_energy(self, positions, atomic_numbers):
        mol = gto_loc.Mole()
        mol.atom = [(element, pos) for element, pos in zip(atomic_numbers, positions)]
        mol.basis    = 'ccecpccpvdz'
        mol.unit     = 'A'
        mol.ecp      = 'ccecp'
        mol.charge   = 0
        mol.spin     = 0
        mol.symmetry = False
        mol.build()
        mf = dft.RKS(mol)
        mf.xc = 'pbe'
        mf.conv_tol = 1e-10
        energy = mf.kernel() * convert_energy # need in eV for ASE
        return energy

    def compute_forces(self, positions, atomic_numbers):
        mol = gto_loc.Mole()
        mol.atom = [(element, pos) for element, pos in zip(atomic_numbers, positions)]
        mol.basis    = 'ccecpccpvdz'
        mol.unit     = 'A'
        mol.ecp      = 'ccecp'
        mol.charge   = 0
        mol.spin     = 0
        mol.symmetry = False
        mol.build()
        mf = dft.RKS(mol)
        mf.xc = 'pbe'
        mf.conv_tol = 1e-10
        mf.kernel()
        mf_grad = grad.RKS(mf)
        forces = mf_grad.kernel() * convert_forces # need in eV/A for ASE
        return -forces #GRI force = -ve of gradient
