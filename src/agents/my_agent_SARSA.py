import numpy as np
import os
from agents.base_agent import BaseAgent



class MyAgent(BaseAgent):
    """
    SARSA agent for the Sailing Challenge (tabular version).
    """

    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.exploration_rate = 0.3

        # Discretization
        self.position_bins = 8
        self.velocity_bins = 4
        self.wind_bins = 8

        self.wind_preview_steps = 2

        # Q-table
        self.q_table = {}

        # SARSA
        self.last_state = None
        self.last_action = None


    def discretize_state(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        wind_flattened = observation[6:]

        grid_size = 32
        x_bin = min(int(x / grid_size * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / grid_size * self.position_bins), self.position_bins - 1)

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

    # action selection
    def act(self, observation, info=None):
        state = self.discretize_state(observation)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)

        if self.np_random.random() < self.exploration_rate:
            action = self.np_random.integers(0, 9)
        else:
            action = np.argmax(self.q_table[state])

        # Store for SARSA update
        self.last_state = state
        self.last_action = action

        return action

    # SARSA learning algorithm
    def learn(self, state, action, reward, next_state):
        """
        SARSA update: Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ]
        with a' chosen by the current policy (ε-greedy).
        """

        # Initialize Q-values
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)

        # Choose next action using current policy
        if self.np_random.random() < self.exploration_rate:
            next_action = self.np_random.integers(0, 9)
        else:
            next_action = np.argmax(self.q_table[next_state])

        # SARSA target
        td_target = reward + self.discount_factor * self.q_table[next_state][next_action]
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
            pickle.dump(self.q_table, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
