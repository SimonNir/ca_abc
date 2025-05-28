# README

## Overview

This document provides an overview of the components and functionality of the codebase.

---

### 1. Potential Functions
- **`muller_brown`**: Computes the original potential.
- **`gaussian_bias`**: Computes a 2D Gaussian bias deposited at each visited point.
- **`total_potential`**: Sums the original potential and all deposited biases.

---

### 2. Environment Class
- **`MullerBrownEnv`**: Simulates the system using Langevin dynamics.
    - At each step, the RL action deposits a new Gaussian bias at the current position.
    - The state is represented as a 10-dimensional vector, including:
        - The potential (and temperature) at the current position.
        - The potential at four neighboring points.
    - The reward is computed as the spread (variance) of visited positions to encourage exploration.

---

### 3. DDPG Agent
- The agent consists of:
    - **Actor**: Maps the state to a 3-dimensional action.
    - **Critic**: Evaluates Q-values.
    - **Target Networks**: Used for stability during training.
- The agent uses:
    - A **replay buffer** to store transitions.
    - **Soft updates** for learning.

---

### 4. Training Loop
- The main training loop:
    1. Resets the environment for each episode.
    2. Collects transitions.
    3. Performs learning updates.

---

This codebase provides a framework for reinforcement learning in a simulated Muller-Brown potential environment.

# README

## Overview

This document provides an overview of the components and functionality of the codebase.

---

### 1. Potential Functions
- **`muller_brown`**: Computes the original potential.
- **`gaussian_bias`**: Computes a 2D Gaussian bias deposited at each visited point.
- **`total_potential`**: Sums the original potential and all deposited biases.

---

### 2. Environment Class
- **`MullerBrownEnv`**: Simulates the system using Langevin dynamics.
    - At each step, the RL action deposits a new Gaussian bias at the current position.
    - The state is represented as a 10-dimensional vector, including:
        - The potential (and temperature) at the current position.
        - The potential at four neighboring points.
    - The reward is computed as the spread (variance) of visited positions to encourage exploration.

---

### 3. DDPG Agent
- The agent consists of:
    - **Actor**: Maps the state to a 3-dimensional action.
    - **Critic**: Evaluates Q-values.
    - **Target Networks**: Used for stability during training.
- The agent uses:
    - A **replay buffer** to store transitions.
    - **Soft updates** for learning.

---

### 4. Training Loop
- The main training loop:
    1. Resets the environment for each episode.
    2. Collects transitions.
    3. Performs learning updates.

---

This codebase provides a framework for reinforcement learning in a simulated Muller-Brown potential environment.
