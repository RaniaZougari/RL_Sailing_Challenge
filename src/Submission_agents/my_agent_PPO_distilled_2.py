"""
Q-Learning Agent for the Sailing Challenge - Trained Model

This file contains a Q-learning agent trained on the sailing environment.
The agent uses a discretized state space and a Q-table for decision making.
"""

import numpy as np
from evaluator.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

        # Q-table
        self.q_table = {}
        self._init_q_table()
    
    def _init_q_table(self):
        """Initialize the Q-table with learned values."""
        self.q_table[(4, 0, 4, 0)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 5, 0)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 5, 1)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 5, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 1, 5, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 2, 5, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 2, 5, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 2, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 2, 6, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 3, 6, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 3, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 3, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 3, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 4, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 4, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 5, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(3, 5, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(3, 6, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(2, 6, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(2, 7, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(1, 7, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(0, 7, 7, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 7, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 7, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 7, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(2, 7, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 7, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 3, 5, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 3, 6, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 4, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 4, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 4, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 4, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 5, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 5, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 6, 7, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 6, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 7, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 1, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 2, 6, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 3, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(2, 7, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 3, 5, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 3, 6, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 3, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(3, 7, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(3, 7, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 4, 7)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 5, 7)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 1, 5, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
        self.q_table[(7, 3, 5, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
        self.q_table[(7, 4, 5, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 5, 5, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 5, 6, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 5, 5, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 5, 6, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 5, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 6, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 6, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 6, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 7, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 7, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 7, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 7, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 7, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 6, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 7, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 6, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 7, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 5, 6)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 5, 6)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 5, 7)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 5, 0)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 5, 1)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 5, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 4, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 4, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 6, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 7, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 7, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 7, 7, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 1, 6)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 4, 6)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 6, 7)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 6, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 4, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 4, 6, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 4, 6, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 5, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 6, 5, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 1, 5, 0)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
        self.q_table[(5, 1, 4, 0)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 1, 4, 1)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 1, 4, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 1, 5, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 2, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 4, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 1, 7)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 2, 7)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 4, 7)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 4, 0)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 4, 1)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 2, 0)] = np.array([0., 0., 0., 0., 1., 0., 0., 0., 0.])
        self.q_table[(6, 2, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 4, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 5, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 6, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 5, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 6, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 6, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 7, 1)] = np.array([0., 0., 0., 0., 0., 1., 0., 0., 0.])
        self.q_table[(7, 7, 6, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 7, 6, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 7, 7, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 7, 6, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 6, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(2, 7, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 1, 6, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 1, 0, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 1, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 1, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 1, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 2, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(3, 2, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(3, 3, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(2, 3, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(2, 4, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(1, 4, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(1, 5, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(0, 5, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(0, 6, 7, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 7, 7, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 7, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 3, 6, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 7, 0, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 0, 7, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 7, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 3, 7, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 4, 7, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 5, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 5, 5, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 5, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 6, 5, 6)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 1, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 2, 5, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
        self.q_table[(6, 2, 4, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
        self.q_table[(7, 2, 4, 1)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 2, 4, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
        self.q_table[(7, 2, 5, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 2, 1, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 2, 0, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 4, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 1, 6, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 1, 6, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 2, 6, 1)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 2, 6, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 3, 5, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 3, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 3, 6, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(1, 7, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 3, 6, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 3, 6, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 2, 7, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 2, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 3, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(3, 4, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(3, 4, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 5, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 7, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 7, 7, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 3, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 3, 5, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 3, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 4, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 6, 5, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 6, 5, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 6, 6, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 6, 6, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 5, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
        self.q_table[(4, 2, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(2, 5, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(2, 5, 5, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(2, 6, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 6, 5, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 7, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 6, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 6, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(1, 7, 7, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 7, 6, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 1, 1, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 1, 0, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 1, 6, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(0, 7, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(0, 7, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 7, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 5, 7, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 7, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 6, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 6, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 2, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 4, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 4, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 5, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 6, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 2, 7, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(1, 6, 7, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 6, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(2, 7, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 7, 5, 6)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 3, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 3, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 4, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 4, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 5, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 6, 5, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 1, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 0, 0, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 0, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(3, 0, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(3, 1, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(2, 2, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(1, 3, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(0, 3, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(0, 4, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(0, 6, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 0, 4, 1)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 0, 5, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
        self.q_table[(5, 6, 5, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 3, 5, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 3, 7, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 7, 1, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 1, 4, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 2, 4, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 2, 4, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 3, 4, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 4, 5, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 5, 5, 1)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 3, 0)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 1, 0)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 2, 6, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 5, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 3, 5, 5)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 4, 6, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 5, 5, 6)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 5, 5, 6)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 1, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
        self.q_table[(7, 7, 0, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 4, 6, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 4, 6, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(1, 7, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 1, 4, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
        self.q_table[(3, 6, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 1, 0, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 1, 7, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(4, 3, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 3, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 5, 6, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 6, 6, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 6, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 6, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 6, 6)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 4, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 2, 4)] = np.array([0., 0., 0., 0., 0., 1., 0., 0., 0.])
        self.q_table[(7, 7, 2, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 1, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(2, 7, 6, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(1, 7, 6, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 3, 7, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(1, 5, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(2, 6, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 2, 5, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 6, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 0, 4, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
        self.q_table[(5, 0, 4, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 1, 5, 3)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(3, 7, 6, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(0, 6, 6, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(1, 6, 5, 4)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 5, 7, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 6, 7, 3)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(5, 7, 7, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 6, 2)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 7, 7, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(7, 7, 1, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 7, 6, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(5, 7, 6, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(4, 7, 6, 1)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(7, 7, 2, 2)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])
        self.q_table[(6, 5, 6, 4)] = np.array([0., 0., 0., 0., 0., 0., 0., 1., 0.])
        self.q_table[(6, 6, 6, 5)] = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0.])

    def discretize_state(self, obs):
        """
        Discrétise l'observation en état.
        obs format: [x, y, vx, vy, wx, wy, wind_field...]
        """
        x, y = obs[0], obs[1]
        vx, vy = obs[2], obs[3]
        wx, wy = obs[4], obs[5]
        
        gx, gy = self.goal_x, self.goal_y

        # Position bins
        xb = min(int(x * self.pos_bins / 32), self.pos_bins - 1)
        yb = min(int(y * self.pos_bins / 32), self.pos_bins - 1)

        # Velocity bin
        vb = 0
        if abs(vx) + abs(vy) > 0.3:
            if abs(vx) > abs(vy):
                vb = 1 if vx > 0 else 2
            else:
                vb = 3 if vy > 0 else 0

        # Wind direction bin
        if abs(wx) > abs(wy):
            wb = 1 if wx > 0 else 2
        else:
            wb = 3 if wy > 0 else 0

        # Distance to goal bin
        dx = gx - x
        dy = gy - y
        d = abs(dx) + abs(dy)
        gb = min(int(d / 10), self.goal_bins - 1)

        return (xb, yb, vb, wb, gb)

    def physics_fallback(self, obs):
        """
        Fallback basé sur la physique si état inconnu.
        """
        x, y = obs[0], obs[1]
        wx, wy = obs[4], obs[5]
        
        gx, gy = self.goal_x, self.goal_y
        dx = gx - x
        dy = gy - y

        # Direction préférée vers le but
        if abs(dx) > abs(dy):
            preferred = 2 if dx > 0 else 6  # East or West
        else:
            preferred = 0 if dy > 0 else 4  # North or South

        # Si vent fort, ajuster
        wind_strength = abs(wx) + abs(wy)
        if wind_strength > 1.5:
            # Éviter d'aller contre le vent fort
            wind_angle = np.arctan2(wy, wx)
            goal_angle = np.arctan2(dy, dx)
            
            # Si angle entre but et vent < 45°, tacker
            relative_angle = abs(((goal_angle - wind_angle + np.pi) % (2*np.pi)) - np.pi)
            if relative_angle < np.radians(45):
                # Tack perpendiculairement au vent
                if abs(wx) > abs(wy):
                    return 0 if dy > 0 else 4  # Nord ou Sud
                else:
                    return 2 if dx > 0 else 6  # Est ou Ouest

        return preferred

    def act(self, observation, info=None):
        """
        Sélection d'action basée sur la Q-table.
        """
        state = self.discretize_state(observation)
        if state in self.q_table:
            q_values = self.q_table[state]
            action = int(np.argmax(q_values))
            return action
        
        # Fallback pour états non vus
        return self.physics_fallback(observation)

    def reset(self):
        """Reset entre épisodes."""
        pass

    def seed(self, seed=None):
        """Set the random seed for reproducibility."""
        self.np_random = np.random.default_rng(seed)

