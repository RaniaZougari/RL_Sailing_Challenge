"""
Agent combining Q-Learning with physical sailing rules for improved navigation.
"""

import numpy as np
from agents.base_agent import BaseAgent
from src.sailing_physics import calculate_sailing_efficiency


class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

        # Learning parameters
        self.learning_rate = 0.15   # alpha
        self.discount_factor = 0.99  # gamma
        # Epsilon
        self.exploration_rate = 0.05

        # Discretization parameters
        self.position_bins = 10
        self.velocity_bins = 5
        self.wind_bins = 8
        self.wind_preview_steps = 3
        self.grid_size = 32
        
        # Q-table
        self.q_table = {}
        self.q_init_high = 10.0
        
        # State for learning
        self.last_efficiency = 0.0

    def _action_to_direction(self, action):
        """Convert action index to direction vector."""
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1),
            (0, 0)
        ]
        return np.array(directions[action])


    def discretize_state(self, observation):
        """Discretize the continuous observation into a tuple state."""
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        wind_flattened = observation[6:]

        x_bin = min(int(x / self.grid_size * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / self.grid_size * self.position_bins), self.position_bins - 1)

        v_mag = np.sqrt(vx**2 + vy**2)
        if v_mag < 0.1:
            v_bin = 0
        else:
            v_dir = np.arctan2(vy, vx)
            v_bin = int(((v_dir + np.pi) / (2 * np.pi) * (self.velocity_bins - 1)) + 1) % self.velocity_bins

        wind_angle = np.arctan2(wy, wx)
        wind_bin = int(((wind_angle + np.pi) / (2 * np.pi)) * self.wind_bins) % self.wind_bins

        wind_grid = wind_flattened.reshape(32, 32, 2)
        gx, gy = int(x), int(y)

        preview_offsets = [
            (-self.wind_preview_steps, 0),
            (self.wind_preview_steps, 0),
            (0, self.wind_preview_steps),
            (0, -self.wind_preview_steps),
            (-self.wind_preview_steps, self.wind_preview_steps),
            (-self.wind_preview_steps, -self.wind_preview_steps),
            (self.wind_preview_steps, self.wind_preview_steps),
            (self.wind_preview_steps, -self.wind_preview_steps),
        ]

        preview_bins = []
        for dx, dy in preview_offsets:
            xp = np.clip(gx + dx, 0, 31)
            yp = np.clip(gy + dy, 0, 31)
            wxp, wyp = wind_grid[xp, yp]
            ang = np.arctan2(wyp, wxp)
            preview_bins.append(
                int(((ang + np.pi) / (2 * np.pi)) * self.wind_bins) % self.wind_bins
            )

        return (x_bin, y_bin, v_bin, wind_bin, *preview_bins)

    def act(self, observation, info=None):
        """
        Select an action using epsilon-greedy strategy combined with physical sailing rules.
        """
        state = self.discretize_state(observation)

        if state not in self.q_table:
            # Initialize with random values instead of constant
            self.q_table[state] = self.np_random.random(9) * self.q_init_high

        # --- Physics-based action filtering ---
        wx, wy = observation[4], observation[5]
        wind_mag = np.sqrt(wx**2 + wy**2)
        
        valid_actions = []
        action_vectors = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ] # Actions 0-7
        
        # Threshold for no-go zone (approx 45 degrees)
        # Cos(135 deg) = -0.707. 
        # If cos(theta) < -0.707, it means theta > 135 or theta < -135 (towards opposite of wind)
        no_go_threshold = -0.707

        for i in range(8):
            ax, ay = action_vectors[i]
            # Normalize action vector (actions 1,3,5,7 have length sqrt(2))
            a_mag = np.sqrt(ax**2 + ay**2)
            
            if wind_mag > 0:
                # Dot product divided by magnitudes = cos(theta)
                cos_theta = (ax * wx + ay * wy) / (a_mag * wind_mag)
                
                # Check if we are sailing INTO the wind (against it)
                # Wind blows FROM source TO destination.
                # Disallow if boat vector opposes wind vector
                if cos_theta >= no_go_threshold:
                    valid_actions.append(i)
            else:
                # If no wind, all directions valid
                valid_actions.append(i)
                
        valid_actions.append(8) # Stay is always valid

        # --- Epsilon-greedy ---
        if self.np_random.random() < self.exploration_rate:
            # Exploration: random valid action
            action = self.np_random.choice(valid_actions)
        else:
            # Exploitation: best valid action
            q_values = self.q_table[state].copy()
            # Mask invalid actions with -infinity
            mask = np.full(9, -np.inf)
            mask[valid_actions] = 0
            masked_q_values = q_values + mask
            
            action = np.nanargmax(masked_q_values)

            action = np.nanargmax(masked_q_values)

        # Calculate efficiency of the chosen action to use in learning
        if action < 8: # If moving
            action_vec = self._action_to_direction(action)
            # Normalize
            action_vec = action_vec / np.linalg.norm(action_vec)
            
            # Wind vector from observation
            wind_vec = np.array([wx, wy])
            if wind_mag > 0:
                wind_vec = wind_vec / wind_mag
                
            self.last_efficiency = calculate_sailing_efficiency(action_vec, wind_vec)
        else:
            self.last_efficiency = 0.0

        return action

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = self.np_random.random(9) * self.q_init_high
        if next_state not in self.q_table:
            self.q_table[next_state] = self.np_random.random(9) * self.q_init_high

        # classic bellman
        # Add efficiency bonus to reward
        efficiency_bonus = self.last_efficiency * 0.5
        shaped_reward = reward + efficiency_bonus
        
        td_target = shaped_reward + self.discount_factor * np.nanmax(self.q_table[next_state])
        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.learning_rate * td_error


    def reset(self):
        self.last_state = None
        self.last_action = None

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                'q_table': self.q_table,
                'exploration_rate': self.exploration_rate
            }, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.exploration_rate = data.get('exploration_rate', 0.01)