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
    """
    Compute the Muller-Brown potential at (x, y) using a 4-term expansion.
    The parameters below are taken from the original formulation.
    x, y can be scalars or numpy arrays.
    """
    # Parameters for 4 terms
    A = np.array([-200, -100, -170, 15])
    a = np.array([-1, -1, -6.5, 0.7])
    b = np.array([0, 0, 11, 0.6])
    c = np.array([-10, -10, -6.5, 0.7])
    x0 = np.array([1, 0, -0.5, -1])
    y0 = np.array([0, 0.5, 1.5, 1])
    
    V = 0.0
    for i in range(4):
        V += A[i] * np.exp(a[i]*(x - x0[i])**2 + b[i]*(x - x0[i])*(y - y0[i]) + c[i]*(y - y0[i])**2)
    return V

def gaussian_bias(x, y, center, sigma_x, sigma_y, rho, height=1.0):
    """
    Compute the value of a 2D Gaussian bias function at position (x,y) given:
      center: (cx, cy) the center where bias was deposited,
      sigma_x, sigma_y: standard deviations,
      rho: correlation coefficient.
    """
    cx, cy = center
    dx = x - cx
    dy = y - cy
    
    # Construct covariance matrix (ensuring positive definiteness)
    # Sigma = [[sigma_x^2, rho*sigma_x*sigma_y],
    #          [rho*sigma_x*sigma_y, sigma_y^2]]
    # Its determinant:
    det = sigma_x**2 * sigma_y**2 * (1 - rho**2 + 1e-6)
    # Inverse of Sigma:
    inv11 = sigma_y**2 / det
    inv22 = sigma_x**2 / det
    inv12 = -rho * sigma_x * sigma_y / det
    
    Q = inv11 * dx**2 + 2*inv12 * dx * dy + inv22 * dy**2
    return height * np.exp(-0.5 * Q)

def total_potential(x, y, bias_list):
    """
    Compute the total potential at (x, y) by summing the Muller-Brown potential
    and all Gaussian biases deposited so far.
    """
    V = muller_brown(x, y)
    for bias in bias_list:
        center, sigma_x, sigma_y, rho, height = bias
        V += gaussian_bias(x, y, center, sigma_x, sigma_y, rho, height)
    return V

###############################
# Environment: MullerBrownEnv
###############################

class MullerBrownEnv:
    def __init__(self, dt=0.01, gamma=1.0, T=0.1, max_steps=200, neighbor_delta=0.1):
        self.dt = dt          # time step for Langevin dynamics
        self.gamma = gamma    # friction coefficient
        self.T = T            # temperature
        self.max_steps = max_steps
        self.neighbor_delta = neighbor_delta  # distance to sample neighbor points
        
        self.noise_std = np.sqrt(2 * T * dt / gamma)
        self.reset()
    
    def reset(self):
        self.current_pos = np.array([0.0, 0.0])
        self.visited_positions = [self.current_pos.copy()]
        self.bias_list = []   # store deposited biases as tuples: (center, sigma_x, sigma_y, rho, height)
        self.step_count = 0
        return self.get_observation()
    
    def get_observation(self):
        """
        Construct the observation consisting of the potential at:
          - the current position,
          - up, down, left, and right positions (neighbors)
        Also include the temperature information (here appended five times).
        The observation is a 10-dimensional numpy array.
        """
        x, y = self.current_pos
        delta = self.neighbor_delta
        points = [
            (x, y),
            (x, y + delta),
            (x, y - delta),
            (x - delta, y),
            (x + delta, y)
        ]
        obs = []
        for pt in points:
            V_pt = total_potential(pt[0], pt[1], self.bias_list)
            obs.append(V_pt)
        # Append the temperature (T) for each point (could be useful if T varied spatially)
        obs += [self.T] * len(points)
        return np.array(obs, dtype=np.float32)
    
    def compute_numerical_force(self, pos, eps=1e-5):
        """
        Compute the force at position pos by finite differences.
        F = -grad(V_total)
        """
        x, y = pos
        V_center = total_potential(x, y, self.bias_list)
        V_x_plus = total_potential(x + eps, y, self.bias_list)
        V_x_minus = total_potential(x - eps, y, self.bias_list)
        V_y_plus = total_potential(x, y + eps, self.bias_list)
        V_y_minus = total_potential(x, y - eps, self.bias_list)
        
        dV_dx = (V_x_plus - V_x_minus) / (2 * eps)
        dV_dy = (V_y_plus - V_y_minus) / (2 * eps)
        force = -np.array([dV_dx, dV_dy])
        return force
    
    
    def step(self, action):
        """
        Take one environment step.
        The input action is a numpy array of shape (3, ) corresponding to raw output from the RL agent:
          [raw_sigma_x, raw_sigma_y, raw_rho]
        We transform these outputs to valid parameters:
          sigma_x, sigma_y > 0 (using softplus)
          rho in (-1,1) (using tanh)
        Then we deposit a new Gaussian bias at the current position with these parameters.
        After depositing the bias, we compute the force (from the total potential) and then take a
        Langevin dynamics step.
        """
        self.step_count += 1
        
        # Transform raw action outputs
        # Here we use numpy’s exponential for softplus (a rough approximation)
        sigma_x = np.log(1 + np.exp(action[0]))
        sigma_y = np.log(1 + np.exp(action[1]))
        rho = np.tanh(action[2])
        
        # Deposit the new Gaussian bias at the current position
        bias = (self.current_pos.copy(), sigma_x, sigma_y, rho, 1.0)  # fixed height = 1.0
        self.bias_list.append(bias)
        
        # Compute force at current position from the total (biased) potential
        force = self.compute_numerical_force(self.current_pos)
        
        # Langevin dynamics step: deterministic + stochastic update
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=2)
        self.current_pos = self.current_pos + (force / self.gamma) * self.dt + noise
        
        self.visited_positions.append(self.current_pos.copy())
        
        # Compute reward as the sum of variances along x and y from visited positions
        positions = np.array(self.visited_positions)
        var_x = np.var(positions[:, 0])
        var_y = np.var(positions[:, 1])
        reward = (var_x + var_y)  # SDN wants to make negative - want to minimize variance 
        
        # Get next observation
        obs = self.get_observation()
        
        # Episode termination when max_steps reached
        done = self.step_count >= self.max_steps
        
        return obs, reward, done

###############################
# DDPG Agent Components
###############################

# Actor Network: maps observation to action (3 numbers).
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # No activation on final layer; we will apply transformations later.
        x = self.fc3(x)
        return x

# Critic Network: evaluates Q(state, action)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# Simple Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Store transitions as tuples
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# DDPG Agent wrapper
class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 64
        self.action_dim = action_dim
    
    def select_action(self, state, noise_std=0.1):
        """
        Select action given state (as a numpy array). Returns a numpy array.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor)
        self.actor.train()
        action = action.cpu().data.numpy().flatten()
        
        # Add exploration noise
        action += np.random.normal(0, noise_std, size=self.action_dim)
        return action
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch of transitions
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
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update of target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

###############################
# Main Training Loop
###############################

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    env = MullerBrownEnv()
    state_dim = 10  # 5 potentials + 5 temperatures
    action_dim = 3  # sigma_x, sigma_y, and raw rho
    agent = DDPGAgent(state_dim, action_dim, device='cpu')
    
    num_episodes = 100
    max_steps = env.max_steps
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        for step in range(max_steps):
            action = agent.select_action(state, noise_std=0.1)
            next_state, reward, done = env.step(action)
            
            # Store the transition in the replay buffer.
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            # Update agent
            agent.update()
            
            if done:
                break
                
        print(f"Episode {episode+1:03d}, Reward: {episode_reward:.3f}")
        
if __name__ == "__main__":
    main()
