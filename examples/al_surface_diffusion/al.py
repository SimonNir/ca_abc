import numpy as np
import time, hashlib
from ase import Atoms
from ase.build import fcc100, add_adsorbate
from ase.calculators.eam import EAM
from ase.constraints import FixAtoms
from ase.io import write
from ca_abc.potentials import ASEPotential
from ca_abc.optimizers import FIREOptimizer, ASEOptimizer, ScipyOptimizer

def unique_id():
    """Generate a unique run ID using timestamp + short hash."""
    now = str(time.time()).encode("utf-8")
    return time.strftime("%Y%m%d_%H%M%S") + "_" + hashlib.sha1(now).hexdigest()[:6]

class AlSurfaceDiffusion(ASEPotential):
    """
    Al adatom diffusion on Al(100) surface using EAM potential.
    This system recreates the benchmark from Kushima et al. 2009.
    """
    
    def __init__(self, 
                 surface_size=(7, 7),  # Surface unit cells
                 layers=6,              # Number of Al layers
                 vacuum=15.0,           # Vacuum in Angstroms
                 fix_bottom_layers=2,   # Number of bottom layers to fix
                 lattice_constant=None, # Will use database value if None
                 eam_potential='Al99.eam.alloy'):  # EAM potential file
        
        self.surface_size = surface_size
        self.layers = layers
        self.vacuum = vacuum
        self.fix_bottom_layers = fix_bottom_layers
        self.lattice_constant = lattice_constant
        self.eam_potential = eam_potential
        
        # Build the system
        atoms = self._build_surface_with_adatom()
        
        # Set up EAM calculator
        calc = EAM(potential=eam_potential)
        atoms.calc = calc
        
        # Initialize ASESubsetPES
        super().__init__(atoms, calc)
        
    def _build_surface_with_adatom(self):
        """Build Al(100) surface slab with one adatom"""

        # Create Al(100) surface slab
        if self.lattice_constant:
            slab = fcc100('Al', size=(*self.surface_size, self.layers), 
                        a=self.lattice_constant, vacuum=self.vacuum)
        else:
            slab = fcc100('Al', size=(*self.surface_size, self.layers), 
                        vacuum=self.vacuum)

        # Compute center position in xy plane
        cell = slab.get_cell()
        center_xy = [cell[0, 0] / 2, cell[1, 1] / 2]

        # Add adatom
        add_adsorbate(slab, 'Al', height=2.5, position=center_xy)

        # Fix bottom layers
        if self.fix_bottom_layers > 0:
            z_coords = slab.positions[:, 2]
            z_sorted = np.sort(np.unique(z_coords))

            fix_indices = []
            for i in range(self.fix_bottom_layers):
                layer_z = z_sorted[i]
                layer_atoms = np.where(np.abs(z_coords - layer_z) < 0.1)[0]
                fix_indices.extend(layer_atoms)

            slab.set_constraint(FixAtoms(indices=fix_indices))

        return slab
    
    def known_barriers(self):
        """Known activation barriers for Al/Al(100) diffusion (from literature)"""
        return {
            'exchange_mechanism': 0.229,  # eV
        }

    def get_adatom_position(self, free_position):
        """Return the 3D position of the adatom from free coords."""
        full_pos = self._reconstruct_full_position(free_position)
        z_coords = full_pos[:, 2]
        adatom_index = np.argmax(z_coords)
        return full_pos[adatom_index]
    
    def get_biased_atom_indices(self, z_tol=0.5, verbose=False):
        """Find the adatom + nearest top-layer atom in free_atoms."""
        free_pos = self.free_atoms.get_positions()

        adatom_index = int(np.argmax(free_pos[:, 2]))
        adatom_pos = free_pos[adatom_index]

        non_adatom_indices = [i for i in range(len(self.free_atoms)) if i != adatom_index]
        non_adatom_pos = free_pos[non_adatom_indices]
        max_surface_z = np.max(non_adatom_pos[:, 2])

        top_layer_mask = (non_adatom_pos[:, 2] >= max_surface_z - z_tol)
        top_layer_indices = np.array(non_adatom_indices)[top_layer_mask]

        if len(top_layer_indices) == 0:
            raise RuntimeError("No top-layer surface atoms found among free_atoms")

        ad_xy = adatom_pos[:2]
        top_layer_xy = free_pos[top_layer_indices, :2]
        dxy = np.linalg.norm(top_layer_xy - ad_xy, axis=1)
        surface_atom_index = int(top_layer_indices[np.argmin(dxy)])

        return [adatom_index, surface_atom_index]
        

def run_al_benchmark():
    from ca_abc.ca_abc import CurvatureAdaptiveABC
    from ca_abc.analysis import ABCAnalysis

    al_system = AlSurfaceDiffusion(
        surface_size=(7, 7),
        layers=6,
        vacuum=15.0,
        fix_bottom_layers=2
    )
    
    print(f"Created Al(100) surface with {len(al_system.atoms)} atoms")
    
    biased_indices = [al_system.get_biased_atom_indices()[1]]
    print(f"Biasing atoms with indices: {biased_indices}")
    print(f"Their full positions are: \n{al_system.atoms.positions[biased_indices]}")

    abc = CurvatureAdaptiveABC(
        potential=al_system,
        curvature_method="None",
        dump_every=50,
        perturb_type="stochastic",
        default_perturbation_size=0.01,
        scale_perturb_by_curvature=True,
        bias_height_type="fixed", 
        default_bias_height=0.005,
        bias_covariance_type="fixed",
        default_bias_covariance=0.6,
        use_ema_adaptive_scaling=True,
        conservative_ema_delta=True,
        max_descent_steps=1000,
        descent_convergence_threshold=1e-5,
        struc_uniqueness_rmsd_threshold=0.1,
        energy_diff_threshold=0.01,
        biased_atom_indices=biased_indices
    )

    optimizer = ASEOptimizer(abc, optimizer_class='BFGS', maxstep=0.05)
    abc.run(
        optimizer=optimizer,
        max_iterations=300,
        verbose=True,
        stopping_minima_number=2,
        verbose_opt=True,
    )

    analyzer = ABCAnalysis(abc)

    run_id = unique_id()
    analyzer.plot_diagnostics(save_plots=True,
                              filename=f"al_surface_diagnostics_{run_id}.png")
    
    return abc, al_system, run_id


if __name__ == "__main__":
    abc, system, run_id = run_al_benchmark()
    print("\nAl surface diffusion benchmark completed!")

    template = system.atoms

    minima_structures = []
    for i, x in enumerate(abc.minima):
        idx = abc.min_indices[i]
        energy = abc.unbiased_energies[idx]
        atoms = template.copy()
        atoms.set_positions(system._reconstruct_full_position(x))
        atoms.info['energy'] = energy
        minima_structures.append(atoms)

    minima_file = f"al_surface_minima_{run_id}.xyz"
    write(minima_file, minima_structures)
    print(f"Saved {len(minima_structures)} visited minima to '{minima_file}'")

    saddle_structures = []
    for i, x in enumerate(abc.saddles):
        idx = abc.saddle_indices[i]
        energy = abc.unbiased_energies[idx]
        atoms = template.copy()
        atoms.set_positions(system._reconstruct_full_position(x))
        atoms.info['energy'] = energy
        saddle_structures.append(atoms)

    saddles_file = f"al_surface_saddles_{run_id}.xyz"
    write(saddles_file, saddle_structures)
    print(f"Saved {len(saddle_structures)} visited saddles to '{saddles_file}'")

    structures = []
    for i, x in enumerate(abc.trajectory):
        energy = abc.unbiased_energies[i]
        atoms = template.copy()
        atoms.set_positions(system._reconstruct_full_position(x))
        atoms.info['energy'] = energy
        structures.append(atoms)

    traj_file = f"al_surface_traj_{run_id}.xyz"
    write(traj_file, structures)
    print(f"Saved full traj to '{traj_file}'")
