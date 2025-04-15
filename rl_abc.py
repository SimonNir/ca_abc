import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

###############################
# Potential and Bias Functions
###############################

def muller_brown(x, y):
    """Compute the Muller-Brown potential with numerical safeguards."""
    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])
    
    V = 0.0
    for i in range(4):
        # Clip exponent arguments to prevent overflow
        exponent = a[i]*(x - x0[i])**2 + b[i]*(x - x0[i])*(y - y0[i]) + c[i]*(y - y0[i])**2
        exponent = np.clip(exponent, -100, 100)  # Prevent overflow
        V += A[i] * np.exp(exponent)
    return V

def gaussian_bias(x, y, center, sigma_x, sigma_y, rho, height=1.0):
    """Compute 2D Gaussian bias with numerical safeguards."""
    cx, cy = center
    dx = x - cx
    dy = y - cy
    
    # Constrain parameters to reasonable ranges
    sigma_x = np.clip(sigma_x, 0.1, 5.0)
    sigma_y = np.clip(sigma_y, 0.1, 5.0)
    rho = np.clip(rho, -0.99, 0.99)
    
    # Safely compute quadratic form
    det = sigma_x**2 * sigma_y**2 * (1 - rho**2 + 1e-6)
    inv11 = sigma_y**2 / det
    inv22 = sigma_x**2 / det
    inv12 = -rho * sigma_x * sigma_y / det
    
    # Clip dx and dy to prevent overflow
    dx = np.clip(dx, -10, 10)
    dy = np.clip(dy, -10, 10)
    
    Q = inv11 * dx**2 + 2*inv12 * dx * dy + inv22 * dy**2
    Q = np.clip(Q, -100, 100)  # Prevent overflow
    return height * np.exp(-0.5 * Q)

def total_potential(x, y, bias_list):
    """Total potential with numerical safeguards."""
    V = muller_brown(x, y)
    for bias in bias_list:
        V += gaussian_bias(x, y, *bias)
    return np.clip(V, -1e6, 1e6)  # Clip final potential

###############################
# Environment: MullerBrownEnv
###############################

class MullerBrownEnv:
    def __init__(self, dt=0.01, gamma=1.0, T=0.1, max_steps=200, neighbor_delta=0.1):
        self.dt = dt
        self.gamma = gamma
        self.T = T
        self.max_steps = max_steps
        self.neighbor_delta = neighbor_delta
        self.noise_std = np.sqrt(2 * T * dt / gamma)
        self.reset()
    
    def reset(self):
        self.current_pos = np.array([0.0, 0.0])  # Start at origin
        self.visited_positions = [self.current_pos.copy()]
        self.bias_list = []
        self.step_count = 0
        return self.get_observation()
    
    def get_observation(self):
        """Construct observation with numerical safeguards."""
        x, y = self.current_pos
        delta = self.neighbor_delta
        points = [
            (x, y),
            (x, y + delta),
            (x, y - delta),
            (x - delta, y),
            (x + delta, y)
        ]
        
        # Compute potentials with clipping
        potentials = []
        for pt in points:
            V = total_potential(pt[0], pt[1], self.bias_list)
            potentials.append(V)
        potentials = np.array(potentials, dtype=np.float32)
        
        # Safe normalization
        if np.allclose(potentials, potentials[0]):
            potentials_normalized = np.zeros_like(potentials)
        else:
            potentials_normalized = (potentials - np.mean(potentials)) / (np.std(potentials) + 1e-6)
            potentials_normalized = np.clip(potentials_normalized, -10, 10)
        
        # Include additional state information
        obs = np.concatenate([
            potentials_normalized,
            np.array([self.T] * 5),
            self.current_pos / 2.0  # Normalized position
        ])
        return obs
    
    def compute_numerical_force(self, pos, eps=1e-5):
        """Compute force with numerical safeguards."""
        x, y = pos
        V = total_potential(x, y, self.bias_list)
        V_x_plus = total_potential(x + eps, y, self.bias_list)
        V_x_minus = total_potential(x - eps, y, self.bias_list)
        V_y_plus = total_potential(x, y + eps, self.bias_list)
        V_y_minus = total_potential(x, y - eps, self.bias_list)
        
        dV_dx = (V_x_plus - V_x_minus) / (2 * eps)
        dV_dy = (V_y_plus - V_y_minus) / (2 * eps)
        force = -np.array([dV_dx, dV_dy])
        return np.clip(force, -100, 100)  # Clip force
    
    def step(self, action):
        """Environment step with numerical safeguards."""
        self.step_count += 1
        
        # Transform and constrain action parameters
        sigma_x = np.clip(np.log(1 + np.exp(action[0])) + 0.1, 0.1, 5.0)
        sigma_y = np.clip(np.log(1 + np.exp(action[1])) + 0.1, 0.1, 5.0)
        rho = np.clip(np.tanh(action[2]), -0.99, 0.99)
        
        # Deposit new bias
        self.bias_list.append((self.current_pos.copy(), sigma_x, sigma_y, rho, 1.0))
        
        # Compute force and update position
        force = self.compute_numerical_force(self.current_pos)
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=2)
        self.current_pos += (force / self.gamma) * self.dt + noise
        self.current_pos = np.clip(self.current_pos, -5, 5)  # Keep position bounded
        self.visited_positions.append(self.current_pos.copy())
        
        # Safe reward calculation
        positions = np.array(self.visited_positions)
        var_x = np.var(positions[:, 0])
        var_y = np.var(positions[:, 1])
        
        if var_x + var_y < 1e-6:
            reward = -0.1
        else:
            reward = np.log(var_x + var_y + 1e-6)
        
        obs = self.get_observation()
        done = self.step_count >= self.max_steps
        return obs, float(reward), done  # Ensure reward is Python float

###############################
# DDPG Agent Components
###############################

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        # Initialize final layer with small weights
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
    
    def forward(self, state):
        x = torch.tanh(self.fc1(state))  # Additional nonlinearity
        x = torch.tanh(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Constrain actions to [-1, 1]

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # Initialize final layer with small weights
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*batch))
    
    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, actor_lr=1e-4, critic_lr=1e-3, 
                 gamma=0.99, tau=0.005, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 128
        self.action_dim = action_dim
    
    def select_action(self, state, noise_std=0.1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        action += np.random.normal(0, noise_std, size=self.action_dim)
        return np.clip(action, -1, 1)
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Critic update
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
        
        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft updates
        with torch.no_grad():
            for t, s in zip(self.actor_target.parameters(), self.actor.parameters()):
                t.data.mul_(1 - self.tau).add_(s.data, alpha=self.tau)
            for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
                t.data.mul_(1 - self.tau).add_(s.data, alpha=self.tau)

###############################
# Main Training Loop
###############################

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    env = MullerBrownEnv(max_steps=200)
    agent = DDPGAgent(state_dim=12, action_dim=3, device='cpu')
    
    for episode in range(100):
        state = env.reset()
        episode_reward = 0
        
        for step in range(env.max_steps):
            action = agent.select_action(state, noise_std=0.1)
            next_state, reward, done = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            agent.update()
            
            if done:
                break
                
        print(f"Episode {episode+1:3d} | Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    main()