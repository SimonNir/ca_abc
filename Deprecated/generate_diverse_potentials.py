import numpy as np
import matplotlib.pyplot as plt

###############################
# Diverse Potential Energy Surfaces for Training
###############################

def muller_brown(x, y):
    """Original Muller-Brown potential."""
    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])
    
    V = 0.0
    for i in range(4):
        exponent = a[i]*(x - x0[i])**2 + b[i]*(x - x0[i])*(y - y0[i]) + c[i]*(y - y0[i])**2
        exponent = np.clip(exponent, -100, 100)
        V += A[i] * np.exp(exponent)
    return V

def three_hole_potential(x, y):
    """Three-well potential with different barrier heights."""
    # Three Gaussian wells
    V1 = -50 * np.exp(-((x + 1)**2 + y**2) / 0.5)
    V2 = -40 * np.exp(-((x - 1)**2 + y**2) / 0.3)  
    V3 = -60 * np.exp(-(x**2 + (y - 1.5)**2) / 0.4)
    
    # Connecting barriers
    barrier1 = 30 * np.exp(-(x**2 + (y - 0.7)**2) / 0.2)
    barrier2 = 25 * np.exp(-((x - 0.5)**2 + (y + 0.5)**2) / 0.15)
    
    return V1 + V2 + V3 + barrier1 + barrier2

def rastrigin_potential(x, y, A=20):
    """Modified 2D Rastrigin - many local minima."""
    n = 2
    return A * n + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

def rosenbrock_potential(x, y, a=1, b=100):
    """2D Rosenbrock function - narrow curved valley."""
    return (a - x)**2 + b * (y - x**2)**2

def mexican_hat_potential(x, y):
    """Mexican hat potential - ring of minima."""
    r = np.sqrt(x**2 + y**2)
    return 10 * (r - 1.5)**2 * np.exp(-r**2/2) - 20

def double_well_1d_extended(x, y):
    """Double well in x, harmonic in y - different barrier shapes."""
    V_x = (x**2 - 1)**2  # Double well along x
    V_y = 0.5 * y**2     # Harmonic along y
    return V_x + V_y

def lennard_jones_cluster(x, y):
    """Simple 2D approximation of LJ cluster potential."""
    # Multiple LJ-like wells at different positions
    positions = [(-1, -1), (1, -1), (0, 1), (-1, 1), (1, 1)]
    V = 0
    for x0, y0 in positions:
        r = np.sqrt((x - x0)**2 + (y - y0)**2) + 0.1  # Avoid singularity
        V += 4 * ((1/r)**12 - (1/r)**6)
    return np.clip(V, -100, 100)

def randomized_gaussian_mixture(x, y, seed=None):
    """Randomly generated Gaussian mixture - different each time."""
    if seed is not None:
        np.random.seed(seed)
    
    n_gaussians = np.random.randint(3, 8)
    V = 0
    
    for i in range(n_gaussians):
        # Random center
        cx = np.random.uniform(-2, 2)
        cy = np.random.uniform(-2, 2)
        
        # Random amplitude (mix of wells and barriers)
        A = np.random.uniform(-80, 40)
        
        # Random width
        sigma = np.random.uniform(0.2, 0.8)
        
        V += A * np.exp(-((x - cx)**2 + (y - cy)**2) / sigma**2)
    
    return V

###############################
# Potential Manager Class
###############################

class PotentialManager:
    """Manages diverse potentials for training."""
    
    def __init__(self):
        self.potentials = {
            'muller_brown': muller_brown,
            'three_hole': three_hole_potential,
            'rastrigin': rastrigin_potential,
            'rosenbrock': rosenbrock_potential,
            'mexican_hat': mexican_hat_potential,
            'double_well': double_well_1d_extended,
            'lennard_jones': lennard_jones_cluster,
        }
        
    def get_random_potential(self):
        """Sample a random potential for training."""
        name = np.random.choice(list(self.potentials.keys()))
        return name, self.potentials[name]
    
    def get_randomized_gaussian_mixture(self, seed=None):
        """Generate a new random Gaussian mixture."""
        return 'random_mixture', lambda x, y: randomized_gaussian_mixture(x, y, seed)
    
    def curriculum_learning_sequence(self):
        """Return potentials in order of increasing complexity."""
        return [
            ('double_well', double_well_1d_extended),      # Simple
            ('three_hole', three_hole_potential),          # Medium
            ('muller_brown', muller_brown),                 # Medium-hard
            ('mexican_hat', mexican_hat_potential),         # Complex topology
            ('rastrigin', rastrigin_potential),             # Many minima
            ('lennard_jones', lennard_jones_cluster),       # Realistic physics
        ]

###############################
# Local Feature Extraction (Your Hessian Idea!)
###############################

def compute_local_features(potential_func, x, y, delta=0.1):
    """
    Compute local features that are transferable across potentials.
    This is your Hessian-based idea - extract geometric properties!
    """
    
    # First and second derivatives (Hessian matrix)
    eps = 1e-5
    
    # Gradient
    dV_dx = (potential_func(x + eps, y) - potential_func(x - eps, y)) / (2 * eps)
    dV_dy = (potential_func(x, y + eps) - potential_func(x, y - eps)) / (2 * eps)
    
    # Hessian
    d2V_dx2 = (potential_func(x + eps, y) - 2*potential_func(x, y) + potential_func(x - eps, y)) / eps**2
    d2V_dy2 = (potential_func(x, y + eps) - 2*potential_func(x, y) + potential_func(x, y - eps)) / eps**2
    d2V_dxdy = (potential_func(x + eps, y + eps) - potential_func(x + eps, y - eps) - 
                potential_func(x - eps, y + eps) + potential_func(x - eps, y - eps)) / (4 * eps**2)
    
    # Hessian eigenvalues (local curvature)
    H = np.array([[d2V_dx2, d2V_dxdy], [d2V_dxdy, d2V_dy2]])
    eigenvals = np.linalg.eigvals(H)
    
    # Local features (scale-invariant and transferable)
    features = {
        'gradient_magnitude': np.sqrt(dV_dx**2 + dV_dy**2),
        'gradient_direction': np.arctan2(dV_dy, dV_dx),
        'curvature_max': np.max(eigenvals),
        'curvature_min': np.min(eigenvals),
        'curvature_ratio': np.max(eigenvals) / (np.min(eigenvals) + 1e-6),
        'is_minimum': (eigenvals > 0).all(),
        'is_maximum': (eigenvals < 0).all(), 
        'is_saddle': (eigenvals[0] * eigenvals[1] < 0),
        'local_roughness': np.std([
            potential_func(x + delta*np.cos(theta), y + delta*np.sin(theta))
            for theta in np.linspace(0, 2*np.pi, 8)
        ])
    }
    
    return features

def extract_observation_features(potential_func, x, y, neighbor_delta=0.1):
    """
    Enhanced observation that includes transferable geometric features.
    This replaces your simple potential sampling with richer information.
    """
    
    # Original neighbor potentials
    points = [
        (x, y),
        (x, y + neighbor_delta),
        (x, y - neighbor_delta), 
        (x - neighbor_delta, y),
        (x + neighbor_delta, y)
    ]
    
    potentials = [potential_func(px, py) for px, py in points]
    potentials = np.array(potentials)
    
    # Normalize potentials
    if np.std(potentials) > 1e-6:
        potentials_norm = (potentials - np.mean(potentials)) / np.std(potentials)
    else:
        potentials_norm = np.zeros_like(potentials)
    
    # Local geometric features
    local_features = compute_local_features(potential_func, x, y, neighbor_delta)
    
    # Combine into transferable observation
    obs = np.concatenate([
        potentials_norm,  # Local potential landscape
        [local_features['gradient_magnitude']],
        [local_features['curvature_max']],
        [local_features['curvature_min']], 
        [local_features['curvature_ratio']],
        [float(local_features['is_minimum'])],
        [float(local_features['is_saddle'])],
        [local_features['local_roughness']]
    ])
    
    return obs

###############################
# Training Curriculum
###############################

class CurriculumTrainer:
    """Implements curriculum learning across diverse potentials."""
    
    def __init__(self, agent):
        self.agent = agent
        self.potential_manager = PotentialManager()
        self.current_difficulty = 0
        
    def train_episode(self, potential_name=None):
        """Train one episode on a specific or random potential."""
        
        if potential_name is None:
            # Sample random potential
            if np.random.random() < 0.3:  # 30% chance of random mixture
                pot_name, potential_func = self.potential_manager.get_randomized_gaussian_mixture()
            else:
                pot_name, potential_func = self.potential_manager.get_random_potential()
        else:
            pot_name = potential_name
            potential_func = self.potential_manager.potentials[potential_name]
        
        # Create environment with this potential
        env = DiversePotentialEnv(potential_func)
        
        # Run episode
        state = env.reset()
        episode_reward = 0
        
        for step in range(env.max_steps):
            action = self.agent.select_action(state)
            next_state, reward, done = env.step(action)
            self.agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            self.agent.update()
            
            if done:
                break
        
        return episode_reward, pot_name

###############################
# Environment with Diverse Potentials  
###############################

class DiversePotentialEnv:
    """Environment that can work with any potential function."""
    
    def __init__(self, potential_func, dt=0.01, gamma=1.0, T=0.1, max_steps=200):
        self.potential_func = potential_func
        self.dt = dt
        self.gamma = gamma
        self.T = T
        self.max_steps = max_steps
        self.noise_std = np.sqrt(2 * T * dt / gamma)
        self.reset()
    
    def reset(self):
        self.current_pos = np.array([0.0, 0.0])
        self.visited_positions = [self.current_pos.copy()] 
        self.bias_list = []
        self.step_count = 0
        return self.get_observation()
    
    def get_observation(self):
        """Get transferable observation using geometric features."""
        x, y = self.current_pos
        return extract_observation_features(self.potential_func, x, y)
    
    def compute_force(self, pos, eps=1e-5):
        """Compute force from current potential + biases."""
        x, y = pos
        
        # Force from main potential
        V_x_plus = self.potential_func(x + eps, y)
        V_x_minus = self.potential_func(x - eps, y)
        V_y_plus = self.potential_func(x, y + eps)
        V_y_minus = self.potential_func(x, y - eps)
        
        dV_dx = (V_x_plus - V_x_minus) / (2 * eps)
        dV_dy = (V_y_plus - V_y_minus) / (2 * eps)
        
        # Add forces from biases
        for center, sigma_x, sigma_y, rho, height in self.bias_list:
            cx, cy = center
            dx, dy = x - cx, y - cy
            
            # Gaussian bias force
            bias_force_x = height * (dx / sigma_x**2) * np.exp(-0.5 * ((dx/sigma_x)**2 + (dy/sigma_y)**2))
            bias_force_y = height * (dy / sigma_y**2) * np.exp(-0.5 * ((dx/sigma_x)**2 + (dy/sigma_y)**2))
            
            dV_dx += bias_force_x
            dV_dy += bias_force_y
        
        force = -np.array([dV_dx, dV_dy])
        return np.clip(force, -100, 100)
    
    def step(self, action):
        """Step environment (same as before but works with any potential)."""
        self.step_count += 1
        
        # Process action to bias parameters
        sigma_x = np.clip(np.log(1 + np.exp(action[0])) + 0.1, 0.1, 5.0)
        sigma_y = np.clip(np.log(1 + np.exp(action[1])) + 0.1, 0.1, 5.0)
        rho = np.clip(np.tanh(action[2]), -0.99, 0.99)
        
        # Deposit bias
        self.bias_list.append((self.current_pos.copy(), sigma_x, sigma_y, rho, 1.0))
        
        # Update position
        force = self.compute_force(self.current_pos)
        noise = np.random.normal(0, self.noise_std, size=2)
        self.current_pos += (force / self.gamma) * self.dt + noise
        self.current_pos = np.clip(self.current_pos, -5, 5)
        self.visited_positions.append(self.current_pos.copy())
        
        # Reward based on exploration
        positions = np.array(self.visited_positions)
        var_x = np.var(positions[:, 0])
        var_y = np.var(positions[:, 1])
        reward = np.log(var_x + var_y + 1e-6) if var_x + var_y > 1e-6 else -0.1
        
        obs = self.get_observation()
        done = self.step_count >= self.max_steps
        
        return obs, float(reward), done

###############################
# Visualization
###############################

def visualize_diverse_potentials():
    """Plot all the different potentials for comparison."""
    pm = PotentialManager()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    
    potentials_to_plot = list(pm.potentials.items())
    potentials_to_plot.append(('random_mixture', lambda x, y: randomized_gaussian_mixture(x, y, 42)))
    
    for i, (name, func) in enumerate(potentials_to_plot[:8]):
        Z = np.zeros_like(X)
        for ix in range(len(x)):
            for iy in range(len(y)):
                Z[iy, ix] = func(X[iy, ix], Y[iy, ix])
        
        axes[i].contour(X, Y, Z, levels=20)
        axes[i].set_title(name.replace('_', ' ').title())
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diverse_potentials.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualize_diverse_potentials()