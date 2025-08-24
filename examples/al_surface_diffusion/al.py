import numpy as np
from ase import Atoms
from ase.build import fcc100, add_adsorbate
from ase.calculators.eam import EAM
from ase.constraints import FixAtoms
from ca_abc.potentials import ASEPotential
from ca_abc.optimizers import FIREOptimizer, ASEOptimizer, ScipyOptimizer

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
        
        # Initialize ASESubsetPES (which will automatically identify fixed atoms)
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
        center_xy = [cell[0, 0] / 2, cell[1, 1] / 2]  # center in x and y

        # Add adatom at center position
        add_adsorbate(slab, 'Al', height=2.5, position=center_xy)

        # Fix bottom layers to simulate bulk behavior
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
            'exchange_mechanism': 0.229,  # eV - from Kushima et al.
        }

    def get_adatom_position(self, free_position):
        """
        Extract the 3D position of the adatom from the free position vector.
        Assumes the adatom is the atom with highest z after reconstruction.
        """
        full_pos = self._reconstruct_full_position(free_position)
        z_coords = full_pos[:, 2]
        adatom_index = np.argmax(z_coords)
        return full_pos[adatom_index]
    
    # In your AlSurfaceDiffusion class, add this new method:

    def get_biased_atom_indices(self, z_tol=0.5, verbose=False):
        """
        Finds:
        1. The adatom (highest z among free atoms)
        2. The surface atom in the *top layer of non-adatoms* directly under the adatom (nearest in XY)

        Only considers free atoms. Returns indices relative to self.free_atoms.

        Parameters
        ----------
        z_tol : float
            Atoms within (max_top_z - z_tol) are considered part of the top layer.
        verbose : bool
            If True, print debug info.

        Returns
        -------
        [adatom_index, surface_atom_index]  (indices are for self.free_atoms)
        """
        free_pos = self.free_atoms.get_positions()

        # 1. Identify adatom
        adatom_index = int(np.argmax(free_pos[:, 2]))
        adatom_pos = free_pos[adatom_index]
        if verbose:
            print(f"Adatom index (free_atoms): {adatom_index}, z = {adatom_pos[2]:.3f}")

        # 2. Find top layer of *non-adatom* atoms (in free_atoms)
        non_adatom_indices = [i for i in range(len(self.free_atoms)) if i != adatom_index]
        non_adatom_pos = free_pos[non_adatom_indices]
        max_surface_z = np.max(non_adatom_pos[:, 2])

        # Tolerance filter for top layer
        top_layer_mask = (non_adatom_pos[:, 2] >= max_surface_z - z_tol)
        top_layer_indices = np.array(non_adatom_indices)[top_layer_mask]

        if len(top_layer_indices) == 0:
            raise RuntimeError("No top-layer surface atoms found among free_atoms")

        # 3. Choose the top-layer atom closest in XY to the adatom
        ad_xy = adatom_pos[:2]
        top_layer_xy = free_pos[top_layer_indices, :2]
        dxy = np.linalg.norm(top_layer_xy - ad_xy, axis=1)
        surface_atom_index = int(top_layer_indices[np.argmin(dxy)])

        if verbose:
            print(f"Surface atom index (free_atoms): {surface_atom_index}, z = {free_pos[surface_atom_index, 2]:.3f}")

        return [adatom_index, surface_atom_index]
        

# In your run_al_benchmark function, modify the code like this:

def run_al_benchmark():
    """
    Run the Al surface diffusion benchmark that matches Kushima et al.
    """
    from ca_abc.ca_abc import CurvatureAdaptiveABC
    from ca_abc.optimizers import ASEOptimizer
    
    # Create Al surface system
    al_system = AlSurfaceDiffusion(
        surface_size=(7, 7),
        layers=6,
        vacuum=15.0,
        fix_bottom_layers=2
    )
    
    print(f"Created Al(100) surface with {len(al_system.atoms)} atoms")
    
    # --- New code starts here ---
    # Get the indices for the adatom and the central surface atom
    biased_indices = [al_system.get_biased_atom_indices()[1]]
    # biased_indices = None # 32
    
    # Print the indices for verification
    print(f"Biasing atoms with indices: {biased_indices}")
    print(f"Their full positions are: \n{al_system.atoms.positions[biased_indices]}")

    # Set up CA-ABC with parameters suitable for metallic surfaces
    abc = CurvatureAdaptiveABC(
        potential=al_system,
        curvature_method="None",
        dump_every=30000,
        
        # Perturbation parameters - smaller for metal surfaces
        perturb_type="fixed",
        default_perturbation_size=0.05,  # Angstroms
        scale_perturb_by_curvature=True,
        
        # Bias parameters - tuned for Al surface barriers (~0.23 eV)
        bias_height_type="fixed", 
        default_bias_height=0.05,  # eV
        
        # Covariance - based on Al lattice parameter (~4.05 Å)
        bias_covariance_type="fixed",
        default_bias_covariance=0.5,  # Å²
        
        # Conservative EMA scaling
        use_ema_adaptive_scaling=True,
        conservative_ema_delta=True,
        
        # Convergence criteria
        max_descent_steps=10000,
        descent_convergence_threshold=5e-3,  # eV/Å
        struc_uniqueness_rmsd_threshold=0.1,  # Å
        energy_diff_threshold=0.01,  # eV

        biased_atom_indices=biased_indices
    )

    # Run the simulation
    # optimizer = ASEOptimizer(abc, optimizer_class='BFGS', 
                            #  max_step_size=0.1,
                            #  )
    # optimizer = ScipyOptimizer(abc, method='BFGS')
    optimizer = FIREOptimizer(abc, max_step_size=None)
    abc.run(
        optimizer=optimizer,
        max_iterations=300,
        verbose=True,
        stopping_minima_number=3,  # Stop after finding 2 distinct minima
        verbose_opt=True,
    )

    from ca_abc.analysis import ABCAnalysis
    analyzer = ABCAnalysis(abc)
    analyzer.plot_diagnostics()
    
    return abc, al_system

if __name__ == "__main__":
    abc, system = run_al_benchmark()
    print("\nAl surface diffusion benchmark completed!")

    # print("Biases:", abc.bias_list)

    from ase.io import write

    # Save visited minima to a trajectory file
    template = system.atoms
    minima_structures = []
    for i, x in enumerate(abc.minima):
        idx = abc.min_indices[i]
        energy = abc.unbiased_energies[idx]
        atoms = template.copy()
        atoms.set_positions(system._reconstruct_full_position(x))
        atoms.info['energy'] = energy
        minima_structures.append(atoms)

    write("al_surface_minima.xyz", minima_structures)
    print(f"Saved {len(minima_structures)} visited minima to 'al_surface_minima.xyz'")

    # Save visited saddles with energies
    saddle_structures = []
    for i, x in enumerate(abc.saddles):
        idx = abc.saddle_indices[i]
        energy = abc.unbiased_energies[idx]
        atoms = template.copy()
        atoms.set_positions(system._reconstruct_full_position(x))
        atoms.info['energy'] = energy
        saddle_structures.append(atoms)

    write("al_surface_saddles.xyz", saddle_structures)
    print(f"Saved {len(saddle_structures)} visited saddles to 'al_surface_saddles.xyz'")

    # Save full trajectory with energies
    structures = []
    for i, x in enumerate(abc.trajectory):
        energy = abc.unbiased_energies[i]
        atoms = template.copy()
        atoms.set_positions(system._reconstruct_full_position(x))
        atoms.info['energy'] = energy
        structures.append(atoms)

    write("al_surface_traj.xyz", structures)
    print(f"Saved full traj to 'al_surface_traj.xyz'")



