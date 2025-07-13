import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE

class DoubleWellCalculator(Calculator):
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, a=1.0, b=4.0, biases=None):
        super().__init__()
        self.a = a
        self.b = b
        self.biases = biases if biases is not None else []
    
    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = atoms.positions[0,0]
        # Base potential
        energy = self.a * x**4 - self.b * x**2
        
        # Bias potentials (simple Gaussians to push system away from known minima)
        bias_energy = 0.0
        bias_force = 0.0
        for center, strength, width in self.biases:
            dx = x - center
            # Gaussian bias: strength * exp(-(dx/width)^2)
            bias_energy += strength * np.exp(- (dx/width)**2)
            # Force is negative gradient of bias energy
            bias_force -= bias_energy * (-2 * dx / width**2)  # chain rule
        
        total_energy = energy + bias_energy
        total_force = -(4 * self.a * x**3 - 2 * self.b * x) + bias_force
        
        self.results['energy'] = total_energy
        self.results['forces'] = np.array([[total_force, 0.0, 0.0]])

# ABC Algorithm
def autonomous_basin_climbing(atoms, calc, n_basins=3, fmax=0.01, max_steps=200):
    found_minima = []
    biases = []
    traj_x_all = []
    traj_e_all = []
    
    for basin_idx in range(n_basins):
        print(f"\nStarting basin search #{basin_idx+1}")
        # Attach trajectory recorder
        traj_x = []
        traj_e = []
        
        def record(step=None):
            x = atoms.positions[0,0]
            e = atoms.get_potential_energy()
            traj_x.append(x)
            traj_e.append(e)
            # Optional: print(f"Step {step}: x={x:.4f}, E={e:.4f}")
        
        opt = FIRE(atoms, logfile=None)
        opt.attach(record)
        opt.run(fmax=fmax, steps=max_steps)
        
        # Record final min position and energy
        min_x = atoms.positions[0,0]
        min_e = atoms.get_potential_energy()
        print(f"Found minimum at x = {min_x:.4f}, E = {min_e:.4f}")
        
        found_minima.append((min_x, min_e))
        traj_x_all.extend(traj_x)
        traj_e_all.extend(traj_e)
        
        # Add a bias potential at the found minimum to push next search away
        bias_strength = 2.0  # Adjust strength to push out of basin
        bias_width = 0.3     # Width of Gaussian bias
        biases.append((min_x, bias_strength, bias_width))
        calc.biases = biases  # Update calculator biases
        
        # Restart atoms near new random position (away from known minima)
        # Here just randomly perturb position away from found minimum
        while True:
            new_x = np.random.uniform(-2.0, 2.0)
            # Ensure new starting pos is away from existing minima by > bias_width
            if all(abs(new_x - m[0]) > bias_width for m in found_minima):
                break
        atoms.positions[0,0] = new_x
        print(f"Restarting at x = {new_x:.4f} for next basin search")
    
    return found_minima, traj_x_all, traj_e_all

# Setup atoms and calculator
atoms = Atoms('H', positions=[[0.5,0,0]])
calc = DoubleWellCalculator(a=1.0, b=4.0)
atoms.calc = calc

# Run Autonomous Basin Climbing
found_minima, traj_x, traj_e = autonomous_basin_climbing(atoms, calc, n_basins=2, fmax=0.01, max_steps=200)

# Plot PES + trajectory
x_vals = np.linspace(-2, 2, 400)
base_pes = calc.a * x_vals**4 - calc.b * x_vals**2

plt.figure(figsize=(8,5))
plt.plot(x_vals, base_pes, 'k-', label='Base PES')
plt.scatter(*zip(*found_minima), color='red', s=100, label='Found minima')
plt.plot(traj_x, traj_e, 'b-', alpha=0.7, label='ABC trajectory')
plt.xlabel('x coordinate')
plt.ylabel('Energy')
plt.title('Autonomous Basin Climbing on 1D Double Well')
plt.legend()
plt.grid(True)
plt.show()
