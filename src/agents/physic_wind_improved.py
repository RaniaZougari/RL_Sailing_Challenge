"""
Improved Agent combining Q-Learning with physical sailing rules.
Enhancements:
- Goal direction in state representation
- Distance-based reward shaping (progress reward)
- Increased efficiency bonus
- Epsilon decay during training
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
        
        # Epsilon - start higher for better exploration during training
        self.exploration_rate = 0.1
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # Discretization parameters
        self.position_bins = 10
        self.velocity_bins = 5
        self.wind_bins = 8
        self.goal_direction_bins = 8  # NEW: bins for direction to goal
        self.wind_preview_steps = 3
        self.grid_size = 32
        
        # Goal position (top center of grid)
        self.goal_x = 16
        self.goal_y = 31
        
        # Q-table
        self.q_table = {}
        self.q_init_high = 10.0
        
        # State for learning
        self.last_efficiency = 0.0
        self.last_position = None  # Track position for progress reward

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

        # NEW: Direction to goal (helps agent know where to go)
        goal_dx = self.goal_x - x
        goal_dy = self.goal_y - y
        goal_angle = np.arctan2(goal_dy, goal_dx)
        goal_dir_bin = int(((goal_angle + np.pi) / (2 * np.pi)) * self.goal_direction_bins) % self.goal_direction_bins

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

        # Include goal_dir_bin in state
        return (x_bin, y_bin, v_bin, wind_bin, goal_dir_bin, *preview_bins)

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
        no_go_threshold = -0.707

        for i in range(8):
            ax, ay = action_vectors[i]
            a_mag = np.sqrt(ax**2 + ay**2)
            
            if wind_mag > 0:
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

        # Calculate efficiency of the chosen action
        if action < 8:
            action_vec = self._action_to_direction(action)
            action_vec = action_vec / np.linalg.norm(action_vec)
            
            wind_vec = np.array([wx, wy])
            if wind_mag > 0:
                wind_vec = wind_vec / wind_mag
                
            self.last_efficiency = calculate_sailing_efficiency(action_vec, wind_vec)
        else:
            self.last_efficiency = 0.0

        # Store current position for progress reward calculation
        self.last_position = (observation[0], observation[1])

        return action

    def learn(self, state, action, reward, next_state, next_observation=None):
        if state not in self.q_table:
            self.q_table[state] = self.np_random.random(9) * self.q_init_high
        if next_state not in self.q_table:
            self.q_table[next_state] = self.np_random.random(9) * self.q_init_high

        # Classic Bellman update
        td_target = reward + self.discount_factor * np.nanmax(self.q_table[next_state])
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        
        # Epsilon decay after each learning step
        if self.exploration_rate > self.epsilon_min:
            self.exploration_rate *= self.epsilon_decay

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_position = None
        self.last_efficiency = 0.0

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
