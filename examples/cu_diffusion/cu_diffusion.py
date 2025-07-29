import numpy as np
from ase import Atoms
from ase.calculators.eam import EAM
from ca_abc.potentials import CanonicalASEPES
from ca_abc.ca_abc import CurvatureAdaptiveABC
from ca_abc.optimizers import FIREOptimizer

# 1. Set up the Cu system
def create_cu_system(num_atoms=10, lattice_constant=3.615):
    """Create a Cu system with vacancies for diffusion study"""
    # Create FCC lattice
    cu = Atoms('Cu' + str(num_atoms), 
              positions=np.random.rand(num_atoms, 3) * lattice_constant * 2,
              cell=[lattice_constant*3]*3,
              pbc=True)
    
    # Remove one atom to create a vacancy
    del cu[0]
    
    return cu

# 2. Load EAM potential for Cu
def load_cu_eam_potential(potential_file='Cu.eam.alloy'):
    """Load EAM potential for Cu"""
    try:
        return EAM(potential=potential_file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find EAM potential file '{potential_file}'. "
            "Please download from NIST Interatomic Potentials Repository: "
            "https://www.ctcms.nist.gov/potentials/"
        )

# 3. Set up the canonical PES
def create_canonical_pes(num_atoms=9, k_soft=10):
    """Create canonical PES for Cu diffusion"""
    # Create system with vacancy
    cu_system = create_cu_system(num_atoms=num_atoms+1)  # +1 because we delete one
    
    # Load EAM potential
    eam_calc = load_cu_eam_potential()
    
    # Create canonical PES
    return CanonicalASEPES(cu_system, k_soft=k_soft)

# 4. Configure and run ABC
def run_cu_diffusion_simulation():
    # Create the potential
    pes = create_canonical_pes(num_atoms=9)  # 9 atoms + 1 vacancy
    
    # Configure ABC parameters
    abc = CurvatureAdaptiveABC(
        potential=pes,
        starting_position=pes.default_starting_position(),
        dump_every=100,
        dump_folder="cu_diffusion_data",
        
        # Perturbation parameters
        perturb_type="adaptive",
        scale_perturb_by_curvature=True,
        default_perturbation_size=0.1,
        min_perturbation_size=0.01,
        max_perturbation_size=0.5,
        
        # Biasing parameters
        bias_height_type="adaptive",
        default_bias_height=1.0,
        min_bias_height=0.1,
        max_bias_height=5.0,
        
        bias_covariance_type="adaptive",
        default_bias_covariance=0.1,
        min_bias_covariance=0.01,
        max_bias_covariance=1.0,
        
        # Optimization parameters
        descent_convergence_threshold=1e-3,
        max_descent_steps=1000,
        max_acceptable_force_mag=10.0
    )
    
    # Create optimizer (FIRE is generally robust for this)
    optimizer = FIREOptimizer(abc, dt=0.01, max_step_size=0.1)
    
    # Run the simulation
    abc.run(
        optimizer=optimizer,
        max_iterations=500,
        verbose=True,
        save_summary=True,
        summary_file="cu_diffusion_summary.txt",
        stopping_minima_number=5  # Stop after finding 5 distinct minima
    )

if __name__ == "__main__":
    run_cu_diffusion_simulation()