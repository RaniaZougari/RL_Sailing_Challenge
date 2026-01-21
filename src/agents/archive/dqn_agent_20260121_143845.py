"""
Deep Q-Network (DQN) Agent for Sailing Challenge
Uses neural network for function approximation instead of Q-table
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from agents.base_agent import BaseAgent

class QNetwork(nn.Module):
    """Neural network for Q-value approximation"""
    
    def __init__(self, state_size=9, action_size=9, hidden_sizes=[128, 64, 32]):
        super(QNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class MyAgentTrained(BaseAgent):
    """DQN Agent for sailing challenge"""
    
    def __init__(self):
        self.np_random = np.random.default_rng()
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.state_size = 9
        self.action_size = 9
        self.q_network = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.batch_size = 64
        self.update_target_every = 100
        self.steps = 0
        
        # Exploration
        self.exploration_rate = 1.0
        self.exploration_rate_min = 0.01
        self.exploration_rate_decay = 0.995
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)
        
        # Normalization constants (will be updated during training)
        self.state_mean = np.zeros(self.state_size)
        self.state_std = np.ones(self.state_size)
    
    def extract_features(self, observation):
        """Extract continuous features from observation"""
        # observation: [x, y, vx, vy, wx, wy, goal_x, goal_y]
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        goal_x, goal_y = 16.0, 32.0  # Fixed goal position
        
        # Calculate derived features
        dx = goal_x - x
        dy = goal_y - y
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_goal = np.arctan2(dy, dx)
        velocity_norm = np.sqrt(vx**2 + vy**2)
        
        # Feature vector
        features = np.array([
            x / 32.0,  # Normalized position
            y / 64.0,
            vx / 5.0,  # Normalized velocity
            vy / 5.0,
            wx / 5.0,  # Normalized wind
            wy / 5.0,
            distance / 50.0,  # Normalized distance
            np.sin(angle_to_goal),  # Angle as sin/cos for continuity
            np.cos(angle_to_goal)
        ], dtype=np.float32)
        
        return features
    
    def discretize_state(self, observation):
        """Compatibility method - DQN uses continuous states directly"""
        return observation  # Return as-is, will be processed in act() and learn()
    
    def normalize_state(self, state):
        """Normalize state using running statistics"""
        return (state - self.state_mean) / (self.state_std + 1e-8)
    
    def act(self, observation):
        """Select action using epsilon-greedy policy"""
        # Epsilon-greedy exploration
        if self.np_random.random() < self.epsilon:
            return self.np_random.integers(0, self.action_size)
        
        # Extract and normalize features
        state = self.extract_features(observation)
        state = self.normalize_state(state)
        
        # Get Q-values from network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def learn(self, state, action, reward, next_state, next_action):
        """Update network using DQN algorithm"""
        # Extract features
        state_features = self.extract_features(state)
        next_state_features = self.extract_features(next_state)
        
        # Update normalization statistics (running average)
        alpha = 0.01
        self.state_mean = (1 - alpha) * self.state_mean + alpha * state_features
        self.state_std = (1 - alpha) * self.state_std + alpha * np.abs(state_features - self.state_mean)
        
        # Reward shaping
        shaped_reward = reward - 5  # Time penalty
        
        # Distance-based bonus
        dx = 16.0 - state[0]
        dy = 32.0 - state[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 5:
            shaped_reward += 10.0
        elif distance < 10:
            shaped_reward += 5.0
        elif distance < 20:
            shaped_reward += 2.0
        elif distance < 30:
            shaped_reward += 0.5
        
        # Velocity bonus
        velocity = np.sqrt(state[2]**2 + state[3]**2)
        if velocity >= 1.0:
            shaped_reward += 0.5
        
        # Store transition
        done = (reward > 0)  # Episode ends when goal is reached
        self.memory.push(state_features, action, shaped_reward, next_state_features, done)
        
        # Train if enough samples
        if len(self.memory) >= self.batch_size:
            self._train_step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _train_step(self):
        """Perform one training step"""
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Normalize states
        states = (states - self.state_mean) / (self.state_std + 1e-8)
        next_states = (next_states - self.state_mean) / (self.state_std + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
    
    def decay(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset(self):
        """Reset for new scenario - keep network but reset exploration"""
        # Don't reset epsilon - let it continue decaying for convergence
        pass
    
    def seed(self, seed=None):
        """Set random seed"""
        self.np_random = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
    
    def save(self, path):
        """Save the model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.state_mean = checkpoint['state_mean']
        self.state_std = checkpoint['state_std']
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
