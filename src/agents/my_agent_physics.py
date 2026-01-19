"""
Agent combining Q-Learning with physical sailing rules for improved navigation.
"""

import numpy as np
from agents.base_agent import BaseAgent


class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

        # Learning parameters
        self.learning_rate = 0.15
        self.discount_factor = 0.99
        self.exploration_rate = 0.4
        self.exploration_decay = 0.997
        self.min_exploration = 0.01

        # Discretization parameters
        self.position_bins = 10
        self.velocity_bins = 5
        self.wind_bins = 8
        self.wind_preview_steps = 3
        self.grid_size = 32

        # Physical sailing parameters
        self.NO_GO_ANGLE = np.radians(45)      # No-Go Zone
        self.BEAM_REACH_ANGLE = np.radians(90) # Optimal sailing angle
        self.optimal_efficiency = 1.0
        
        # Q-table
        self.q_table = {}
        self.initial_q_value = 10.0


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
            self.q_table[state] = np.full(9, self.initial_q_value)

        # Epsilon-greedy with physics guidance
        if self.np_random.random() < self.exploration_rate:
            # Exploration: random
            action = self.np_random.integers(0, 9)
        else:
            action = np.nanargmax(self.q_table[state])

        return action

    def learn(self, state, action, reward, next_state):
        """
        SARSA avec bonus de récompense basé sur la physique.
        """
        if state not in self.q_table:
            self.q_table[state] = np.full(9, self.initial_q_value)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.full(9, self.initial_q_value)

        # Choose next action for SARSA
        if self.np_random.random() < self.exploration_rate:
            next_action = self.np_random.integers(0, 9)
        else:
            next_action = np.argmax(self.q_table[next_state])

        # SARSA update standard
        td_target = reward + self.discount_factor * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

        # less exploration over time
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

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