"""
Myagent for the Sailing Challenge
"""


import numpy as np
import os
from agents.base_agent import BaseAgent

"""
Hyperparameters:
    - learning_rate: Step size for Q-value updates.
    - discount_factor: Importance of future rewards.
    - exploration_rate: Probability of choosing a random action.

"""


class MyAgent(BaseAgent):
    """
    Q-learning agent for the Sailing Challenge environment.
    """

    def __init__(self):
        super().__init__()
        # Random number generator
        self.np_random = np.random.default_rng()
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.exploration_rate = 0.3
        
        # State discretization parameters
        self.position_bins = 8     # Discretize the grid into 8x8
        self.velocity_bins = 4     # Discretize velocity into 4 bins
        self.wind_bins = 8         # Discretize wind directions into 8 bins

        # Strategy : wind aware Q-learning : wind preview parameters 
        self.wind_preview_steps = 2  # Number of future wind steps to consider
        self.wind_direction_bins = 8    # Discretize wind directions into 8 bins
        
        # Initialize Q-table
        # State space: position_x, position_y, velocity_direction, wind_direction
        # Action space: 9 possible actions
        self.q_table = {}


    def discretize_state(self, observation):
        """Convert continuous observation to discrete state for Q-table lookup."""
        # Extract position, velocity and wind from observation
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        wind_flattened = observation[6:] 
        
        # Discretize position (assume 32x32 grid)
        grid_size = 32
        x_bin = min(int(x / grid_size * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / grid_size * self.position_bins), self.position_bins - 1)
        
        # Discretize velocity direction (ignoring magnitude for simplicity)
        v_magnitude = np.sqrt(vx**2 + vy**2)
        if v_magnitude < 0.1:  # If velocity is very small, consider it as a separate bin
            v_bin = 0
        else:
            v_direction = np.arctan2(vy, vx)  # Range: [-pi, pi]
            v_bin = int(((v_direction + np.pi) / (2 * np.pi) * (self.velocity_bins-1)) + 1) % self.velocity_bins

        # Discretize local wind direction
        local_wind_angle = np.arctan2(wy, wx)  # [-pi, pi]
        local_wind_bin = int(((local_wind_angle + np.pi) / (2 * np.pi)) * self.wind_bins) % self.wind_bins

        # Retrieve future wind directions from the flattened wind array
        wind_grid = wind_flattened.reshape(32, 32, 2)
        gx = int(x)
        gy = int(y)

        # x: row axis , y : column axis
        preview_offsets = [
        (-self.wind_preview_steps, 0),  # N
        (self.wind_preview_steps, 0),   # S
        (0, self.wind_preview_steps),   # E
        (0, -self.wind_preview_steps),  # W
        (-self.wind_preview_steps, self.wind_preview_steps),  # NE
        (-self.wind_preview_steps, -self.wind_preview_steps), # NW
        (self.wind_preview_steps, self.wind_preview_steps),   # SE
        (self.wind_preview_steps, -self.wind_preview_steps),  # SW
        ]

        # Discretize preview wind directions
        preview_winds = []
        for dx, dy in preview_offsets:
            x_prev = np.clip(gx + dx, 0, wind_grid.shape[0] - 1)
            y_prev = np.clip(gy + dy, 0, wind_grid.shape[1] - 1)
            wx, wy = wind_grid[x_prev, y_prev]
            
            angle = np.arctan2(wy, wx)  # [-pi, pi]
            preview_wind_bin= int(((angle + np.pi) / (2 * np.pi)) * self.wind_bins) % self.wind_bins
            preview_winds.append(preview_wind_bin)

        # Return full discrete state
        return (x_bin,y_bin,v_bin,local_wind_bin,*preview_winds)

    
    def act(self, observation, info=None):
        """
        Select an action based on the current observation.
        
        Args:
            observation: A numpy array containing the current observation.
                Format: [x, y, vx, vy, wx, wy] where:
                - (x, y) is the current position
                - (vx, vy) is the current velocity
                - (wx, wy) is the current wind vector
        
        Returns:
            action: An integer in [0, 8] representing the action to take:
                - 0: Move North
                - 1: Move Northeast
                - 2: Move East
                - 3: Move Southeast
                - 4: Move South
                - 5: Move Southwest
                - 6: Move West
                - 7: Move Northwest
                - 8: Stay in place
        """
        """Choose an action using epsilon-greedy policy."""
        # Discretize the state
        state = self.discretize_state(observation)
        
        # Epsilon-greedy action selection
        if self.np_random.random() < self.exploration_rate:
            # Explore: choose a random action
            return self.np_random.integers(0, 9)
        else:
            # Exploit: choose the best action according to Q-table
            if state not in self.q_table:
                # If state not in Q-table, initialize it
                self.q_table[state] = np.zeros(9)
            
            # Return action with highest Q-value
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        """Update Q-table based on observed transition."""
        # Initialize Q-values if states not in table
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)
        
        # Q-learning update
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def reset(self):
        """Reset the agent's internal state between episodes."""
        pass

    def seed(self, seed=None):
        """Set the random seed for reproducibility."""
        self.np_random = np.random.default_rng(seed)

    def save(self, path):
        """Save the Q-table to a file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load(self, path):
        """Load the Q-table from a file."""
        import pickle
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)