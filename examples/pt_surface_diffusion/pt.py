import numpy as np
from ase import Atoms, Atom
from ase.build import fcc111, bulk
from ase.calculators.morse import MorsePotential
from ase.constraints import FixAtoms
from ca_abc.potentials import ASEPotential
from ca_abc.optimizers import FIREOptimizer

class PtSurfaceDiffusion(ASEPotential):
    """
    Pt 7-atom adcluster diffusion on a Pt(111) surface using a Morse potential.
    
    This system recreates the benchmark from Kushima et al. 2009, which is
    based on the experimental and theoretical work of Bassett and Webber (1978).
    The potential is a pairwise Morse potential with parameters fit for Platinum.
    
    The paper describes two activation schemes:
    1.  **Single Atom**: Biasing only a single corner atom of the adcluster, which
        yields a high barrier of 1.695 eV.
    2.  **Collective**: Biasing all 7 adcluster atoms and the top three surface
        layers, which reveals a lower energy concerted migration path with a
        barrier of 0.604 eV.
    """
    
    def __init__(self, 
                 surface_size=(7, 8),      # Surface unit cells (56 atoms/layer)
                 layers=6,                 # Number of Pt layers
                 vacuum=15.0,              # Vacuum in Angstroms
                 fix_bottom_layers=3):     # Number of bottom layers to fix
        
        self.surface_size = surface_size
        self.layers = layers
        self.vacuum = vacuum
        self.fix_bottom_layers = fix_bottom_layers
        
        # Build the system
        atoms = self._build_surface_with_adcluster()
        
        # --- MORSE POTENTIAL SETUP ---
        # Parameters for Pt-Pt interaction are taken from Bassett and Webber,
        # Surface Science 70 (1978) 520-531, Table 2.
        # U_0 (epsilon) = 0.7102 eV
        # alpha = 1.6047 Å⁻¹
        # r_0 = 2.8970 Å
        # The cutoff is set to the relaxation radius (r_rel) from the paper.
        calc = MorsePotential(
            epsilon=0.7102, 
            alpha=1.6047, 
            r0=2.8970, 
            rc=6.25, 
            atomic_number=78 # Atomic number for Platinum
        )
        atoms.calc = calc
        
        # Initialize the base class
        super().__init__(atoms, calc)
        
    def _build_surface_with_adcluster(self):
        """Builds the Pt(111) surface slab with a 7-atom adcluster."""
        
        # Create Pt(111) surface slab
        slab = fcc111('Pt', size=(*self.surface_size, self.layers), vacuum=self.vacuum)

        # Get the theoretical lattice constant for bulk Pt to define the adcluster spacing.
        # The nearest-neighbor distance in an FCC crystal is a/sqrt(2).
        bulk_pt = bulk('Pt', 'fcc')
        a = bulk_pt.cell.lengths()[0]
        bond_length = a / np.sqrt(2)
        
        # Center the cluster on a hollow site
        cell = slab.get_cell()
        center_xy = [cell[0, 0] / 2, cell[1, 1] / 2 - bond_length/3]
        
        # Define positions for the 7-atom cluster
        cluster_positions = [center_xy]
        for i in range(6):
            angle = i * np.pi / 3
            pos = [
                center_xy[0] + bond_length * np.cos(angle),
                center_xy[1] + bond_length * np.sin(angle)
            ]
            cluster_positions.append(pos)

        # --- FIX ---
        # Add the 7 adcluster atoms one by one using ase.Atom (singular).
        z_top_surface = np.max(slab.positions[:, 2])
        for pos_xy in cluster_positions:
            atom_to_add = Atom('Pt', position=(pos_xy[0], pos_xy[1], z_top_surface + 2.0))
            slab.append(atom_to_add)
            
        # Fix bottom layers as described in the paper
        if self.fix_bottom_layers > 0:
            z_coords = slab.positions[:, 2]
            # Find unique z-coordinates of the substrate layers only
            unique_z = np.sort(np.unique(z_coords[:-7])) 
            
            bottom_layer_z_levels = unique_z[:self.fix_bottom_layers]
            
            fix_indices = [
                i for i, pos in enumerate(slab.positions) 
                if pos[2] in bottom_layer_z_levels
            ]
            slab.set_constraint(FixAtoms(indices=fix_indices))

        return slab
        
    def known_barriers(self):
        """Known activation barriers for Pt/Pt(111) diffusion from Kushima et al."""
        return {
            'single_atom_mechanism': 1.695,  # eV
            'collective_mechanism': 0.604,   # eV
        }
        
    def get_biased_atom_indices(self, bias_type='collective'):
        """
        Returns the indices of atoms to be biased for a given mechanism.
        
        Indices are relative to the list of *free* (non-fixed) atoms.

        Parameters
        ----------
        bias_type : str
            - 'single_atom': Biases one corner atom of the adcluster.
            - 'collective': Biases all 7 adcluster atoms and the top 3 surface
              layers (all movable atoms).
        
        Returns
        -------
        list
            A list of integer indices for the atoms to be biased.
        """
        free_pos = self.free_atoms.get_positions()
        
        # The last 7 atoms added are the adcluster atoms. In the free_atoms list,
        # they will also be the last 7.
        adcluster_indices_free = list(range(len(free_pos) - 7, len(free_pos)))
        
        if bias_type == 'single_atom':
            # Find a "corner" atom of the cluster (e.g., the one with max x-value)
            adcluster_pos = free_pos[adcluster_indices_free]
            corner_atom_sub_idx = np.argmax(adcluster_pos[:, 0])
            corner_atom_idx_free = adcluster_indices_free[corner_atom_sub_idx]
            print(f"Biasing a single corner adcluster atom (index: {corner_atom_idx_free}).")
            return [corner_atom_idx_free]
            
        elif bias_type == 'collective':
            # In this system, biasing the cluster and top 3 layers is equivalent
            # to biasing all movable atoms, as specified in the paper.
            all_free_indices = list(range(len(free_pos)))
            print(f"Biasing all {len(all_free_indices)} movable atoms (adcluster + top 3 layers).")
            return all_free_indices
            
        else:
            raise ValueError("bias_type must be 'single_atom' or 'collective'")


def run_pt_benchmark(bias_type='collective'):
    """
    Run the Pt adcluster diffusion benchmark that matches Kushima et al..

    Parameters
    ----------
    bias_type : str
        The biasing scheme to use: 'single_atom' or 'collective'.
    """
    from ca_abc.ca_abc import CurvatureAdaptiveABC

    # Create Pt surface system
    pt_system = PtSurfaceDiffusion()
    
    # Kushima et al. specify 525 degrees of freedom, which means 175 movable atoms.
    # Let's verify our setup matches this.
    num_movable = len(pt_system.free_atoms)
    print(f"Created Pt(111) surface with {len(pt_system.atoms)} total atoms.")
    print(f"Number of movable atoms: {num_movable} (Paper reports 175).")
    assert num_movable == 175, "System setup does not match the paper's DoF."

    # Get the indices for the chosen biasing scheme
    biased_indices = pt_system.get_biased_atom_indices(bias_type=bias_type)
    
    # Set up CA-ABC with parameters suitable for this system
    abc = CurvatureAdaptiveABC(
        potential=pt_system,
        dump_every=100,
        curvature_method='none',
        # Perturbation parameters
        default_perturbation_size=0.02,
        perturb_type="stochastic",
        
        # Bias parameters - tuned for expected barriers (0.6 - 1.7 eV)
        default_bias_height=0.2, # eV
        bias_height_type="fixed",
        default_bias_covariance=0.8, # Å²
        bias_covariance_type="fixed",
        
        # Convergence criteria
        descent_convergence_threshold=1e-2, # eV/Å
        struc_uniqueness_rmsd_threshold=0.15, # Å
        energy_diff_threshold=0.05, # eV
        max_descent_steps=500,

        # Pass the list of atoms to apply the bias potential to
        biased_atom_indices=biased_indices
    )

    # Use the FIRE optimizer
    optimizer = FIREOptimizer(abc, dt=0.05, dt_max=0.1, max_step_size=0.2)
    
    abc.run(
        optimizer=optimizer,
        max_iterations=500,
        verbose=True,
        stopping_minima_number=2, # Stop after finding a second minimum
        verbose_opt=True,
    )
    
    from ca_abc.analysis import ABCAnalysis
    analyzer = ABCAnalysis(abc)
    analyzer.plot_diagnostics(filename=f"pt_surface_diagnostics_{bias_type}_morse.png")
    
    return abc, pt_system


if __name__ == "__main__":
    # --- CHOOSE YOUR BIASING SCHEME HERE ---
    # 'single_atom': Recreates the 1.695 eV barrier
    # 'collective': Recreates the 0.604 eV barrier
    CHOSEN_BIAS_TYPE = 'collective' 
    
    print("-" * 60)
    print(f"Starting Pt adcluster benchmark with '{CHOSEN_BIAS_TYPE}' biasing (Morse Potential).")
    print("-" * 60)
    
    abc_run, system = run_pt_benchmark(bias_type=CHOSEN_BIAS_TYPE)
    
    print(f"\nPt surface diffusion benchmark ('{CHOSEN_BIAS_TYPE}') completed!")
    
    # Save the final trajectory for visualization
    from ase.io import write
    
    structures = []
    template = system.atoms
    for i, x in enumerate(abc_run.trajectory):
        energy = abc_run.unbiased_energies[i]
        atoms = template.copy()
        atoms.set_positions(system._reconstruct_full_position(x))
        atoms.info['energy'] = energy
        structures.append(atoms)

    filename = f"pt_surface_traj_{CHOSEN_BIAS_TYPE}_morse.xyz"
    write(filename, structures)
    print(f"Saved full trajectory to '{filename}'")