"""
Q-Learning Agent for the Sailing Challenge - Trained Model

This file contains a Q-learning agent trained on the sailing environment.
The agent uses a discretized state space and a Q-table for decision making.
"""

import numpy as np
from collections import defaultdict
from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

        # Discrétisation - AFFINÉE
        self.pos_bins = 8
        self.vel_bins = 4
        self.wind_bins = 8
        self.goal_bins = 5

        # Q-table
        self.q_table = {}
        self.initial_q_value = 50.0
        self._init_q_table()
    
    def _init_q_table(self):
        """Initialize the Q-table with learned values."""
        #self.q_table[(6, 0, 50.    , 50.    , 50.    , 50.    ,....])

    def discretize_state(self, obs):
        x, y = obs[0], obs[1]
        vx, vy = obs[2], obs[3]
        wx, wy = obs[4], obs[5]
        gx, gy = obs[-2], obs[-1]

        # --- Position bins ---
        xb = min(int(x * self.pos_bins / 32), self.pos_bins - 1)
        yb = min(int(y * self.pos_bins / 32), self.pos_bins - 1)

        # --- Velocity bin (cheap) ---
        vb = 0
        if abs(vx) + abs(vy) > 0.3:
            if abs(vx) > abs(vy):
                vb = 1 if vx > 0 else 2
            else:
                vb = 3 if vy > 0 else 0

        # --- Wind direction bin ---
        if abs(wx) > abs(wy):
            wb = 1 if wx > 0 else 2
        else:
            wb = 3 if wy > 0 else 0

        # --- Distance to goal bin ---
        dx = gx - x
        dy = gy - y
        d = abs(dx) + abs(dy)
        gb = min(int(d / 10), self.goal_bins - 1)

        return (xb, yb, vb, wb, gb)

    def physics_fallback(self, obs):
        x, y = obs[0], obs[1]
        wx, wy = obs[4], obs[5]
        gx, gy = obs[-2], obs[-1]

        dx = gx - x
        dy = gy - y

        # Prefer moving toward goal
        if abs(dx) > abs(dy):
            preferred = 3 if dx > 0 else 5
        else:
            preferred = 1 if dy > 0 else 7

        # If strong wind blocks, go crosswind
        if abs(wx) > abs(wy):
            return 1 if wx > 0 else 7
        else:
            return preferred

    # ==========================================================
    # ACTION SELECTION (O(1))
    # ==========================================================
    def act(self, observation, info=None):
        state = self.discretize_state(observation)

        # Q-table decision
        action = self.q_table.get(state)

        if action is not None:
            return int(action)

        # Safe fallback
        return self.physics_fallback(observation)

    # ==========================================================
    # REQUIRED METHODS
    # ==========================================================
    def reset(self):
        pass

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)