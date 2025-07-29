import numpy as np
from ase import Atoms
from ase.build import fcc100, add_adsorbate
from ase.calculators.eam import EAM
from ase.constraints import FixAtoms
from ca_abc.potentials import ASEPotentialEnergySurface
from ca_abc.optimizers import FIREOptimizer, ASEOptimizer, ScipyOptimizer

class AlSurfaceDiffusion(ASEPotentialEnergySurface):
    """
    Al adatom diffusion on Al(100) surface using EAM potential.
    This system recreates the benchmark from Kushima et al. 2009.
    """
    
    def __init__(self, 
                 surface_size=(4, 4),  # Surface unit cells
                 layers=4,              # Number of Al layers
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
        
        # Add adatom on top of surface in bridge position (initial guess)
        # Bridge position is between two surface atoms
        add_adsorbate(slab, 'Al', height=2.5, position='bridge')
        
        # Fix bottom layers to simulate bulk behavior
        if self.fix_bottom_layers > 0:
            # Get z-coordinates to identify bottom layers
            z_coords = slab.positions[:, 2]
            z_sorted = np.sort(np.unique(z_coords))
            
            # Find atoms in bottom layers
            fix_indices = []
            for i in range(self.fix_bottom_layers):
                layer_z = z_sorted[i]
                layer_atoms = np.where(np.abs(z_coords - layer_z) < 0.1)[0]
                fix_indices.extend(layer_atoms)
            
            # Apply constraint
            slab.set_constraint(FixAtoms(indices=fix_indices))
        
        return slab
    
    def default_starting_position(self):
        """Return flattened positions as starting point"""
        return self.atoms.positions.flatten()
    
    def known_barriers(self):
        """Known activation barriers for Al/Al(100) diffusion (from literature)"""
        return {
            'exchange_mechanism': 0.23,  # eV - from Kushima et al.
            'hopping_mechanism': 0.65,   # eV - simple hopping (higher barrier)
        }
    
    def get_adatom_position(self, position):
        """Extract adatom position from flattened coordinate vector"""
        pos_3d = position.reshape(-1, 3)
        # Adatom is the last atom (highest z-coordinate initially)
        return pos_3d[-1]
    
    def plot_range(self):
        """Reasonable range for visualization"""
        cell = self.atoms.cell
        return ((0, cell[0, 0]), (0, cell[1, 1]))

def run_al_benchmark():
    """
    Run the Al surface diffusion benchmark that matches Kushima et al.
    """
    from ca_abc.ca_abc import CurvatureAdaptiveABC
    from ca_abc.optimizers import FIREOptimizer
    
    # Create Al surface system
    al_system = AlSurfaceDiffusion(
        surface_size=(4, 4),
        layers=4,
        vacuum=15.0,
        fix_bottom_layers=2
    )
    
    print(f"Created Al(100) surface with {len(al_system.atoms)} atoms")
    print(f"Adatom initial position: {al_system.get_adatom_position(al_system.default_starting_position())}")
    
    # Set up CA-ABC with parameters suitable for metallic surfaces
    abc = CurvatureAdaptiveABC(
        potential=al_system,
        curvature_method="bfgs",
        dump_every=30000,
        
        # Perturbation parameters - smaller for metal surfaces
        perturb_type="fixed",
        default_perturbation_size=0.1,  # Angstroms
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
        max_descent_steps=400,
        descent_convergence_threshold=0.01,  # eV/Å
        struc_uniqueness_rmsd_threshold=0.1,  # Å
    )
    
    # Run the simulation
    optimizer = ASEOptimizer(abc, 'BFGS')
    abc.run(
        optimizer=optimizer,
        max_iterations=300,
        verbose=True,
        stopping_minima_number=2  # Stop after finding 5 distinct minima
    )

    from ca_abc.analysis import ABCAnalysis
    analyzer = ABCAnalysis(abc)
    analyzer.plot_diagnostics()
    
    return abc, al_system

if __name__ == "__main__":
    abc, system = run_al_benchmark()
    print("\nAl surface diffusion benchmark completed!")

    from ase.io import write

    # Save visited minima to a trajectory file
    template = system.atoms
    minima_structures = []
    for x in abc.minima:
        atoms = template.copy()
        atoms.set_positions(x.reshape(-1, 3))
        minima_structures.append(atoms)

    write("al_surface_minima.xyz", minima_structures)
    print(f"Saved {len(minima_structures)} visited minima to 'al_surface_minima.xyz'")

    structures = []
    for x in abc.trajectory:
        atoms = template.copy()
        atoms.set_positions(x.reshape(-1, 3))
        structures.append(atoms)

    write("al_surface_traj.xyz", structures)
    print(f"Saved full traj to 'al_surface_traj.xyz'")


