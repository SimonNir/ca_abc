import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, Tuple, List, Optional
import time
import json

###############################
# Diverse Potential Energy Surfaces
###############################

class PotentialLibrary:
    """Library of diverse potential energy surfaces for training."""
    
    @staticmethod
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
    
    @staticmethod
    def three_hole_potential(x, y):
        """Three-well potential with different barrier heights."""
        V1 = -50 * np.exp(-((x + 1)**2 + y**2) / 0.5)
        V2 = -40 * np.exp(-((x - 1)**2 + y**2) / 0.3)  
        V3 = -60 * np.exp(-(x**2 + (y - 1.5)**2) / 0.4)
        barrier1 = 30 * np.exp(-(x**2 + (y - 0.7)**2) / 0.2)
        barrier2 = 25 * np.exp(-((x - 0.5)**2 + (y + 0.5)**2) / 0.15)
        return V1 + V2 + V3 + barrier1 + barrier2
    
    @staticmethod
    def rastrigin_potential(x, y, A=20):
        """Modified 2D Rastrigin - many local minima."""
        n = 2
        return A * n + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))
    
    @staticmethod
    def rosenbrock_potential(x, y, a=1, b=100):
        """2D Rosenbrock function - narrow curved valley."""
        return (a - x)**2 + b * (y - x**2)**2
    
    @staticmethod
    def mexican_hat_potential(x, y):
        """Mexican hat potential - ring of minima."""
        r = np.sqrt(x**2 + y**2)
        return 10 * (r - 1.5)**2 * np.exp(-r**2/2) - 20
    
    @staticmethod
    def double_well_1d_extended(x, y):
        """Double well in x, harmonic in y."""
        V_x = (x**2 - 1)**2
        V_y = 0.5 * y**2
        return V_x + V_y
    
    @staticmethod
    def lennard_jones_cluster(x, y):
        """Simple 2D approximation of LJ cluster potential."""
        positions = [(-1, -1), (1, -1), (0, 1), (-1, 1), (1, 1)]
        V = 0
        for x0, y0 in positions:
            r = np.sqrt((x - x0)**2 + (y - y0)**2) + 0.1
            V += 4 * ((1/r)**12 - (1/r)**6)
        return np.clip(V, -100, 100)
    
    @staticmethod
    def randomized_gaussian_mixture(x, y, seed=None):
        """Randomly generated Gaussian mixture."""
        if seed is not None:
            np.random.seed(seed)
        
        n_gaussians = np.random.randint(3, 8)
        V = 0
        
        for i in range(n_gaussians):
            cx = np.random.uniform(-2, 2)
            cy = np.random.uniform(-2, 2)
            A = np.random.uniform(-80, 40)
            sigma = np.random.uniform(0.2, 0.8)
            V += A * np.exp(-((x - cx)**2 + (y - cy)**2) / sigma**2)
        
        return V

class PotentialManager:
    """Manages diverse potentials for curriculum learning."""
    
    def __init__(self):
        self.potentials = {
            'muller_brown': PotentialLibrary.muller_brown,
            'three_hole': PotentialLibrary.three_hole_potential,
            'rastrigin': PotentialLibrary.rastrigin_potential,
            'rosenbrock': PotentialLibrary.rosenbrock_potential,
            'mexican_hat': PotentialLibrary.mexican_hat_potential,
            'double_well': PotentialLibrary.double_well_1d_extended,
            'lennard_jones': PotentialLibrary.lennard_jones_cluster,
        }
        
        # Curriculum order (simple to complex)
        self.curriculum_order = [
            'double_well',      # Simple
            'three_hole',       # Medium
            'muller_brown',     # Medium-hard
            'mexican_hat',      # Complex topology
            'rastrigin',        # Many minima
            'lennard_jones',    # Realistic physics
        ]
    
    def get_potential(self, name: str):
        """Get specific potential by name."""
        if name == 'random_mixture':
            seed = np.random.randint(0, 10000)
            return lambda x, y: PotentialLibrary.randomized_gaussian_mixture(x, y, seed)
        return self.potentials[name]
    
    def get_random_potential(self):
        """Sample random potential for training diversity."""
        if np.random.random() < 0.2:  # 20% chance of random mixture
            return 'random_mixture', self.get_potential('random_mixture')
        else:
            name = np.random.choice(list(self.potentials.keys()))
            return name, self.potentials[name]
    
    def get_curriculum_potential(self, difficulty_level: float):
        """Get potential based on curriculum difficulty (0.0 to 1.0)."""
        idx = int(difficulty_level * len(self.curriculum_order))
        idx = min(idx, len(self.curriculum_order) - 1)
        name = self.curriculum_order[idx]
        return name, self.potentials[name]

###############################
# Transferable Feature Extraction
###############################

class GeometricFeatureExtractor:
    """Extracts transferable geometric features from potential landscapes."""
    
    @staticmethod
    def compute_local_features(potential_func, x: float, y: float, eps: float = 1e-5) -> Dict[str, float]:
        """Compute local geometric features that transfer across potentials."""
        
        # Gradient computation
        dV_dx = (potential_func(x + eps, y) - potential_func(x - eps, y)) / (2 * eps)
        dV_dy = (potential_func(x, y + eps) - potential_func(x, y - eps)) / (2 * eps)
        
        # Hessian computation
        d2V_dx2 = (potential_func(x + eps, y) - 2*potential_func(x, y) + potential_func(x - eps, y)) / eps**2
        d2V_dy2 = (potential_func(x, y + eps) - 2*potential_func(x, y) + potential_func(x, y - eps)) / eps**2
        d2V_dxdy = (potential_func(x + eps, y + eps) - potential_func(x + eps, y - eps) - 
                    potential_func(x - eps, y + eps) + potential_func(x - eps, y - eps)) / (4 * eps**2)
        
        # Hessian eigenvalues (principal curvatures)
        H = np.array([[d2V_dx2, d2V_dxdy], [d2V_dxdy, d2V_dy2]])
        eigenvals = np.linalg.eigvals(H)
        eigenvals = np.sort(eigenvals)  # Sort for consistency
        
        # Local roughness (variation in nearby points)
        delta = 0.1
        nearby_values = [
            potential_func(x + delta*np.cos(theta), y + delta*np.sin(theta))
            for theta in np.linspace(0, 2*np.pi, 8, endpoint=False)
        ]
        local_roughness = np.std(nearby_values)
        
        # Compile transferable features
        features = {
            'gradient_magnitude': np.sqrt(dV_dx**2 + dV_dy**2),
            'gradient_direction': np.arctan2(dV_dy, dV_dx),
            'curvature_min': eigenvals[0],
            'curvature_max': eigenvals[1],
            'curvature_mean': np.mean(eigenvals),
            'curvature_ratio': eigenvals[1] / (eigenvals[0] + 1e-6) if eigenvals[0] != 0 else 0,
            'gaussian_curvature': eigenvals[0] * eigenvals[1],
            'mean_curvature': 0.5 * (eigenvals[0] + eigenvals[1]),
            'is_minimum': float((eigenvals > 1e-3).all()),
            'is_maximum': float((eigenvals < -1e-3).all()),
            'is_saddle': float(eigenvals[0] * eigenvals[1] < -1e-6),
            'local_roughness': local_roughness,
        }
        
        return features

    @staticmethod
    def extract_observation_features(potential_func, x: float, y: float, 
                                   neighbor_delta: float = 0.1, 
                                   bias_list: Optional[List] = None) -> np.ndarray:
        """
        Extract comprehensive observation including both local potentials and geometric features.
        This creates a transferable representation that works across different potential types.
        """
        
        # Sample local potential landscape
        points = [
            (x, y),                    # Center
            (x, y + neighbor_delta),   # North
            (x, y - neighbor_delta),   # South
            (x - neighbor_delta, y),   # West
            (x + neighbor_delta, y),   # East
        ]
        
        # Compute total potential (including biases) at each point
        potentials = []
        for px, py in points:
            V = potential_func(px, py)
            if bias_list:
                for center, sigma_x, sigma_y, rho, height in bias_list:
                    V += GeometricFeatureExtractor._gaussian_bias(px, py, center, sigma_x, sigma_y, rho, height)
            potentials.append(V)
        
        potentials = np.array(potentials)
        
        # Normalize potential values to remove absolute scale dependence
        if np.std(potentials) > 1e-6:
            potentials_norm = (potentials - np.mean(potentials)) / np.std(potentials)
        else:
            potentials_norm = np.zeros_like(potentials)
        potentials_norm = np.clip(potentials_norm, -10, 10)
        
        # Extract geometric features
        local_features = GeometricFeatureExtractor.compute_local_features(
            lambda px, py: potential_func(px, py) + sum(
                GeometricFeatureExtractor._gaussian_bias(px, py, *bias) 
                for bias in (bias_list or [])
            ), x, y
        )
        
        # Combine into transferable observation
        feature_values = [
            local_features['gradient_magnitude'],
            local_features['curvature_min'],
            local_features['curvature_max'], 
            local_features['curvature_ratio'],
            local_features['gaussian_curvature'],
            local_features['mean_curvature'],
            local_features['is_minimum'],
            local_features['is_maximum'],
            local_features['is_saddle'],
            local_features['local_roughness']
        ]
        
        # Final observation: [normalized_potentials, geometric_features, position]
        obs = np.concatenate([
            potentials_norm,              # 5 values: local potential landscape
            feature_values,               # 10 values: geometric features  
            [x / 2.0, y / 2.0]          # 2 values: normalized position
        ])
        
        return obs.astype(np.float32)
    
    @staticmethod
    def _gaussian_bias(x, y, center, sigma_x, sigma_y, rho, height=1.0):
        """Helper function to compute Gaussian bias."""
        cx, cy = center
        dx, dy = x - cx, y - cy
        
        sigma_x = np.clip(sigma_x, 0.1, 5.0)
        sigma_y = np.clip(sigma_y, 0.1, 5.0)
        rho = np.clip(rho, -0.99, 0.99)
        
        det = sigma_x**2 * sigma_y**2 * (1 - rho**2 + 1e-6)
        inv11 = sigma_y**2 / det
        inv22 = sigma_x**2 / det
        inv12 = -rho * sigma_x * sigma_y / det
        
        Q = inv11 * dx**2 + 2*inv12 * dx * dy + inv22 * dy**2
        Q = np.clip(Q, -100, 100)
        return height * np.exp(-0.5 * Q)

###############################
# Universal Environment
###############################

class UniversalABCEnvironment:
    """Environment that works with any potential function using transferable features."""
    
    def __init__(self, potential_func, potential_name: str = "unknown",
                 dt: float = 0.01, gamma: float = 1.0, T: float = 0.1, 
                 max_steps: int = 200, neighbor_delta: float = 0.1):
        
        self.potential_func = potential_func
        self.potential_name = potential_name
        self.dt = dt
        self.gamma = gamma
        self.T = T
        self.max_steps = max_steps
        self.neighbor_delta = neighbor_delta
        self.noise_std = np.sqrt(2 * T * dt / gamma)
        
        # State tracking
        self.current_pos = None
        self.visited_positions = []
        self.bias_list = []
        self.step_count = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_pos = np.array([0.0, 0.0])
        self.visited_positions = [self.current_pos.copy()]
        self.bias_list = []
        self.step_count = 0
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get transferable observation using geometric features."""
        x, y = self.current_pos
        return GeometricFeatureExtractor.extract_observation_features(
            self.potential_func, x, y, self.neighbor_delta, self.bias_list
        )
    
    def _compute_total_force(self, pos: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Compute total force including biases."""
        x, y = pos
        
        # Force from main potential
        dV_dx = (self.potential_func(x + eps, y) - self.potential_func(x - eps, y)) / (2 * eps)
        dV_dy = (self.potential_func(x, y + eps) - self.potential_func(x, y - eps)) / (2 * eps)
        
        # Add forces from biases
        for center, sigma_x, sigma_y, rho, height in self.bias_list:
            cx, cy = center
            dx, dy = x - cx, y - cy
            
            # Gaussian bias gradient
            bias_val = GeometricFeatureExtractor._gaussian_bias(x, y, center, sigma_x, sigma_y, rho, height)
            bias_force_x = height * (dx / sigma_x**2) * bias_val
            bias_force_y = height * (dy / sigma_y**2) * bias_val
            
            dV_dx += bias_force_x
            dV_dy += bias_force_y
        
        force = -np.array([dV_dx, dV_dy])
        return np.clip(force, -100, 100)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Execute one environment step."""
        self.step_count += 1
        
        # Transform action to bias parameters
        sigma_x = np.clip(np.log(1 + np.exp(action[0])) + 0.1, 0.1, 3.0)
        sigma_y = np.clip(np.log(1 + np.exp(action[1])) + 0.1, 0.1, 3.0)
        rho = np.clip(np.tanh(action[2]), -0.95, 0.95)
        
        # Deposit new bias at current position
        self.bias_list.append((self.current_pos.copy(), sigma_x, sigma_y, rho, 1.0))
        
        # Update particle position using Langevin dynamics
        force = self._compute_total_force(self.current_pos)
        noise = np.random.normal(0, self.noise_std, size=2)
        self.current_pos += (force / self.gamma) * self.dt + noise
        self.current_pos = np.clip(self.current_pos, -4, 4)  # Keep bounded
        
        self.visited_positions.append(self.current_pos.copy())
        
        # Compute exploration reward
        reward = self._compute_exploration_reward()
        
        # Get new observation
        obs = self._get_observation()
        done = self.step_count >= self.max_steps
        
        return obs, reward, done
    
    def _compute_exploration_reward(self) -> float:
        """Compute reward based on exploration quality."""
        if len(self.visited_positions) < 10:
            return 0.0
            
        positions = np.array(self.visited_positions)
        
        # Reward based on spatial variance (coverage)
        var_x = np.var(positions[:, 0])
        var_y = np.var(positions[:, 1])
        coverage_reward = np.log(var_x + var_y + 1e-6)
        
        # Penalty for too many biases (efficiency)
        efficiency_penalty = -0.01 * len(self.bias_list)
        
        # Bonus for escaping local minima (measured by energy barriers crossed)
        recent_positions = positions[-20:] if len(positions) > 20 else positions
        energy_variance = np.var([self.potential_func(pos[0], pos[1]) for pos in recent_positions])
        barrier_crossing_bonus = 0.1 * np.log(energy_variance + 1e-6)
        
        total_reward = coverage_reward + efficiency_penalty + barrier_crossing_bonus
        return float(total_reward)

###############################
# Universal RL Agent with Transfer Learning
###############################

class UniversalActor(nn.Module):
    """Actor network with potential-type awareness for transfer learning."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Main policy network
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize output layer with small weights for stable learning
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc4.bias, -3e-3, 3e-3)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.fc4(x))  # Actions in [-1, 1]
        return action

class UniversalCritic(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        # Initialize output layer
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc4.bias, -3e-3, 3e-3)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        return q_value

class ReplayBuffer:
    """Experience replay buffer for stable learning."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*batch))
    
    def __len__(self):
        return len(self.buffer)

class UniversalDDPGAgent:
    """Universal DDPG agent that transfers across different potential energy surfaces."""
    
    def __init__(self, state_dim: int = 17, action_dim: int = 3, 
                 hidden_dim: int = 128, actor_lr: float = 1e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.005, device: str = 'cpu'):
        
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        
        # Networks
        self.actor = UniversalActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = UniversalActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = UniversalCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = UniversalCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 64
        
        # Training statistics
        self.training_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'q_values': []
        }
    
    def select_action(self, state: np.ndarray, noise_std: float = 0.1, 
                     exploration: bool = True) -> np.ndarray:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        
        if exploration:
            # Add exploration noise
            action += np.random.normal(0, noise_std, size=self.action_dim)
        
        return np.clip(action, -1, 1)
    
    def update(self):
        """Update actor and critic networks."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update_targets()
        
        # Track statistics
        self.training_stats['actor_loss'].append(actor_loss.item())
        self.training_stats['critic_loss'].append(critic_loss.item())
        self.training_stats['q_values'].append(current_q.mean().item())
    
    def _soft_update_targets(self):
        """Soft update target networks."""
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.mul_(1 - self.tau).add_(param.data, alpha=self.tau)
            
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.mul_(1 - self.tau).add_(param.data, alpha=self.tau)
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']

###############################
# Curriculum Training Manager
###############################

class CurriculumTrainer:
    """Manages curriculum learning across diverse potentials."""
    
    def __init__(self, agent: UniversalDDPGAgent, potential_manager: PotentialManager):
        self.agent = agent
        self.potential_manager = potential_manager
        self.training_history = {
            'episode_rewards': [],
            'potential_names': [],
            'transfer_scores': [],
            'training_times': []
        }
        self.test_potentials = ['rosenbrock']  # Hold-out test set
    
    def train_episode(self, potential_name: Optional[str] = None, 
                     noise_std: float = 0.1) -> Tuple[float, str]:
        """Train one episode on specified or random potential."""
        
        # Select potential
        if potential_name is None:
            pot_name, potential_func = self.potential_manager.get_random_potential()
        else:
            pot_name = potential_name
            potential_func = self.potential_manager.get_potential(potential_name)
        
        # Create environment
        env = UniversalABCEnvironment(potential_func, pot_name, max_steps=150)
        
        # Run episode
        state = env.reset()
        episode_reward = 0.0
        
        for step in range(env.max_steps):
            # Select action
            action = self.agent.select_action(state, noise_std=noise_std)
            
            # Environment step
            next_state, reward, done = env.step(action)
            
            # Store experience
            self.agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update agent
            self.agent.update()
            
            # Prepare for next step
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Record training statistics
        self.training_history['episode_rewards'].append(episode_reward)
        self.training_history['potential_names'].append(pot_name)
        self.training_history['training_times'].append(env.max_steps)
        
        return episode_reward, pot_name
    
    def evaluate_agent(self, n_episodes: int = 10, potential_name: Optional[str] = None) -> Dict[str, float]:
        """Evaluate agent performance on test potentials."""
        if potential_name is None:
            test_potentials = self.test_potentials
        else:
            test_potentials = [potential_name]
        
        results = {}
        
        for pot_name in test_potentials:
            potential_func = self.potential_manager.get_potential(pot_name)
            env = UniversalABCEnvironment(potential_func, pot_name, max_steps=200)
            
            episode_rewards = []
            exploration_metrics = []
            
            for _ in range(n_episodes):
                state = env.reset()
                episode_reward = 0.0
                positions = []
                
                for _ in range(env.max_steps):
                    action = self.agent.select_action(state, noise_std=0.0, exploration=False)
                    state, reward, done = env.step(action)
                    episode_reward += reward
                    positions.append(env.current_pos.copy())
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                
                # Compute exploration metrics
                positions = np.array(positions)
                coverage = np.var(positions[:,0]) + np.var(positions[:,1])
                energy_values = [env.potential_func(p[0], p[1]) for p in positions]
                energy_range = max(energy_values) - min(energy_values)
                exploration_metrics.append((coverage, energy_range))
            
            avg_reward = np.mean(episode_rewards)
            avg_coverage, avg_energy_range = np.mean(exploration_metrics, axis=0)
            
            results[pot_name] = {
                'avg_reward': float(avg_reward),
                'avg_coverage': float(avg_coverage),
                'avg_energy_range': float(avg_energy_range),
                'std_reward': float(np.std(episode_rewards))
            }
        
        return results
    
    def train(self, total_episodes: int = 5000, curriculum: bool = True,
             initial_noise: float = 0.5, noise_decay: float = 0.995):
        """Main training loop with curriculum learning."""
        best_reward = -np.inf
        noise_std = initial_noise
        
        for episode in range(total_episodes):
            # Update curriculum difficulty
            if curriculum:
                difficulty = min(1.0, episode / (total_episodes * 0.7))  # Linear curriculum
                pot_name, _ = self.potential_manager.get_curriculum_potential(difficulty)
            else:
                pot_name = None
            
            # Train episode
            episode_reward, used_pot_name = self.train_episode(pot_name, noise_std)
            
            # Decay exploration noise
            noise_std = max(0.05, noise_std * noise_decay)
            
            # Evaluation and logging
            if episode % 100 == 0:
                eval_results = self.evaluate_agent()
                test_reward = eval_results[self.test_potentials[0]]['avg_reward']
                
                print(f"Episode {episode}: Train Reward = {episode_reward:.1f} "
                      f"(Potential: {used_pot_name}), Test Reward = {test_reward:.1f} "
                      f"(Noise = {noise_std:.3f})")
                
                # Save best model
                if test_reward > best_reward:
                    best_reward = test_reward
                    self.agent.save("best_agent.pth")
                    print("Saved new best model!")
            
            # Plot progress
            if episode % 500 == 0:
                self.plot_training_progress()
    
    def plot_training_progress(self):
        """Plot training statistics."""
        plt.figure(figsize=(15, 5))
        
        # Plot rewards
        plt.subplot(1, 3, 1)
        plt.plot(self.training_history['episode_rewards'])
        plt.title("Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        
        # Plot actor/critic losses
        plt.subplot(1, 3, 2)
        if self.agent.training_stats['actor_loss']:
            plt.plot(self.agent.training_stats['actor_loss'], label="Actor Loss")
            plt.plot(self.agent.training_stats['critic_loss'], label="Critic Loss")
            plt.legend()
            plt.title("Training Losses")
            plt.xlabel("Update Step")
            plt.ylabel("Loss")
        
        # Plot potential distribution
        plt.subplot(1, 3, 3)
        potential_counts = {}
        for name in self.training_history['potential_names']:
            potential_counts[name] = potential_counts.get(name, 0) + 1
        plt.bar(potential_counts.keys(), potential_counts.values())
        plt.title("Potential Distribution")
        plt.xlabel("Potential")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_episode(self, potential_name: str = "muller_brown"):
        """Visualize agent behavior on a potential."""
        potential_func = self.potential_manager.get_potential(potential_name)
        env = UniversalABCEnvironment(potential_func, potential_name, max_steps=200)
        state = env.reset()
        
        # Create potential contour plot
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-1, 2, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = potential_func(X[i,j], Y[i,j])
        
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar(label='Potential Energy')
        plt.title(f"Agent Trajectory on {potential_name} Potential")
        
        # Run episode and plot trajectory
        positions = [env.current_pos.copy()]
        biases = []
        
        for _ in range(env.max_steps):
            action = self.agent.select_action(state, noise_std=0.0, exploration=False)
            state, _, done = env.step(action)
            positions.append(env.current_pos.copy())
            
            # Record biases
            if env.bias_list:
                biases.append(env.bias_list[-1])
            
            if done:
                break
        
        positions = np.array(positions)
        plt.plot(positions[:,0], positions[:,1], 'r-', linewidth=1, label='Trajectory')
        plt.scatter(positions[::10,0], positions[::10,1], c='white', s=30, edgecolors='red', label='Samples')
        
        # Plot biases
        for i, (center, sigma_x, sigma_y, rho, height) in enumerate(biases[::5]):
            cx, cy = center
            angle = 0.5 * np.arctan(2*rho*sigma_x*sigma_y/(sigma_x**2 - sigma_y**2))
            
            # Create ellipse
            from matplotlib.patches import Ellipse
            ellipse = Ellipse((cx, cy), width=sigma_x*2, height=sigma_y*2,
                             angle=angle*np.pi/180, alpha=0.3, color='white')
            plt.gca().add_patch(ellipse)
            if i == 0:
                ellipse.set_label('Biases')
        
        plt.legend()
        plt.xlim(-2, 2)
        plt.ylim(-1, 2)
        plt.show()

###############################
# Main Training Execution
###############################

def main():
    # Initialize components
    potential_manager = PotentialManager()
    agent = UniversalDDPGAgent(state_dim=17, action_dim=3)
    trainer = CurriculumTrainer(agent, potential_manager)
    
    # Train the agent
    print("Starting training...")
    start_time = time.time()
    trainer.train(total_episodes=5000, curriculum=True)
    print(f"Training completed in {time.time()-start_time:.1f} seconds")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    eval_results = trainer.evaluate_agent(n_episodes=20)
    for pot_name, results in eval_results.items():
        print(f"{pot_name}:")
        print(f"  Avg Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Coverage: {results['avg_coverage']:.2f}")
        print(f"  Energy Range: {results['avg_energy_range']:.2f}")
    
    # Visualize performance
    trainer.visualize_episode("muller_brown")
    trainer.visualize_episode("rosenbrock")  # Test potential
    
    # Save final model
    agent.save("final_agent.pth")
    print("Model saved to final_agent.pth")

if __name__ == "__main__":
    main()