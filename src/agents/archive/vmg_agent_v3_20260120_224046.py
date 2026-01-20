"""
VMG Agent v3: Simplified and Amplified
- Relies on native environment reward (progress * 10)
- Removes custom VMG calculation (source of bugs/lag)
- Simplifies state space (removes last_action)
- Adds efficiency bonus and step penalty
"""

import numpy as np
from agents.base_agent import BaseAgent
from src.sailing_physics import calculate_sailing_efficiency

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

        # Learning parameters
        self.learning_rate = 0.15
        self.min_learning_rate = 0.05
        self.lr_decay_rate = 0.999
        
        self.discount_factor = 0.99
        
        # Exploration
        self.exploration_rate = 1      # Start higher to explore new state space
        self.min_exploration = 0.1
        self.eps_decay_rate = 0.997

        # Discretization
        self.position_bins = 10
        self.velocity_bins = 5
        self.wind_bins = 8
        self.grid_size = 32
        
        # Q-table
        self.q_table = {}
        self.q_init_high = 2.0  # Lower initialization
        
        # Tracking
        self.last_efficiency = 0.0
        self.epsilon = 1e-6

    def _action_to_direction(self, action):
        """Convert action index to direction vector."""
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1),
            (0, 0)
        ]
        return np.array(directions[action])

    def discretize_state(self, observation):
        """
        Simplified state discretization.
        REMOVED: last_action (reduces state space by 9x)
        """
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        # Position bins (0-9)
        x_bin = min(int(x / self.grid_size * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / self.grid_size * self.position_bins), self.position_bins - 1)

        # Velocity bin (0=stopped, 1-4=moving direction)
        v_mag = np.sqrt(vx**2 + vy**2)
        if v_mag < 0.1:
            v_bin = 0
        else:
            v_dir = np.arctan2(vy, vx)
            # Map -pi..pi to 0..3
            v_bin = int(((v_dir + np.pi) / (2 * np.pi) * (self.velocity_bins - 1)) + 1) % self.velocity_bins

        # Relative Wind Angle bin (0-7)
        wind_angle = np.arctan2(wy, wx)
        wind_bin = int(((wind_angle + np.pi) / (2 * np.pi)) * self.wind_bins) % self.wind_bins

        return (x_bin, y_bin, v_bin, wind_bin)

    def _calculate_sailing_efficiency_safe(self, action_vec, wind_vec, wind_mag):
        if wind_mag < self.epsilon:
            return 0.0
        
        wind_normalized = wind_vec / wind_mag
        base_efficiency = calculate_sailing_efficiency(action_vec, wind_normalized)
        
        # Scale by wind magnitude
        wind_factor = min(wind_mag / 2.0, 1.0)
        return base_efficiency * wind_factor

    def act(self, observation, info=None):
        state = self.discretize_state(observation)

        # Initialize Q-values if new state
        if state not in self.q_table:
            self.q_table[state] = self.np_random.random(9) * self.q_init_high

        # --- Physics-based action filtering ---
        wx, wy = observation[4], observation[5]
        wind_mag = np.sqrt(wx**2 + wy**2)
        
        valid_actions = []
        action_vectors = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        no_go_threshold = -0.707

        for i in range(8):
            ax, ay = action_vectors[i]
            a_mag = np.sqrt(ax**2 + ay**2)
            
            if wind_mag > self.epsilon:
                # Check if action is in no-go zone relative to wind
                cos_theta = (ax * wx + ay * wy) / (a_mag * wind_mag)
                if cos_theta >= no_go_threshold:
                    valid_actions.append(i)
            else:
                valid_actions.append(i)
                
        valid_actions.append(8) # Stay is always valid

        # --- Epsilon-greedy ---
        if self.np_random.random() < self.exploration_rate:
            action = self.np_random.choice(valid_actions)
        else:
            q_values = self.q_table[state].copy()
            mask = np.full(9, -np.inf)
            mask[valid_actions] = 0
            masked_q_values = q_values + mask
            action = np.nanargmax(masked_q_values)

        # Calculate efficiency for reward shaping
        if action < 8:
            action_vec = self._action_to_direction(action)
            action_vec = action_vec / np.linalg.norm(action_vec)
            wind_vec = np.array([wx, wy])
            self.last_efficiency = self._calculate_sailing_efficiency_safe(
                action_vec, wind_vec, wind_mag
            )
        else:
            self.last_efficiency = 0.0

        return action

    def learn(self, state, action, reward, next_state, next_action=None):
        if state not in self.q_table:
            self.q_table[state] = self.np_random.random(9) * self.q_init_high
        if next_state not in self.q_table:
            self.q_table[next_state] = self.np_random.random(9) * self.q_init_high

        # Recalculate discrete state to be sure
        # (Though 'state' passed in should be correct)

        # --- Reward Shaping ---
        # 1. Base Reward: From Environment (Progress * 10)
        #    - Positive if moving closer to goal
        #    - Negative if moving away
        
        # 2. Efficiency Bonus: Encourage proper alignment with wind
        efficiency_bonus = self.last_efficiency * 0.5
        
        # 3. Step Penalty: Small cost per step to encourage speed
        step_penalty = 0.05

        shaped_reward = reward + efficiency_bonus - step_penalty

        # SARSA update
        td_target = shaped_reward + self.discount_factor * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.learning_rate * td_error

    def reset(self):
        self.last_efficiency = 0.0
        
        self.exploration_rate = max(self.min_exploration, 
                                    self.exploration_rate * self.eps_decay_rate)
        self.learning_rate = max(self.min_learning_rate,
                                 self.learning_rate * self.lr_decay_rate)

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                'q_table': self.q_table,
                'exploration_rate': self.exploration_rate,
                'learning_rate': self.learning_rate
            }, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.exploration_rate = data.get('exploration_rate', 0.01)
            self.learning_rate = data.get('learning_rate', 0.05)
