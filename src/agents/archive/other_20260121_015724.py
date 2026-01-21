import sys
import os
import numpy as np

# Add the src directory to the path
# Get the absolute path to the project root directory (RL_project_sailing)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add both the project root and src directory to Python path
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

# Import the BaseAgent class
from agents.base_agent import BaseAgent
from env_sailing import SailingEnv

class QLearningAgent(BaseAgent):
    """A simple Q-learning agent for the sailing environment using only local information."""
    
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
        
        # Discretization parameters
        self.num_angle_bins = 8
        self.velocity_bins = [0.2, 0.5, 1, 2, 5]
        self.wind_bins = [0.5, 1, 2, 5]
        self.goal_dist_bins = [5, 10, 20, 30, 45]
        
        self.num_actions = 9  # Move this line up before estimate_state_action_space
        
        # Initialize Q-table
        # State space: position_x, position_y, velocity_direction, wind_direction
        # Action space: 9 possible actions
        self.q_table = {}

        
    def discretize_state(self, observation):
        """Convert continuous observation to discrete state for Q-table lookup."""
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        goal_x, goal_y = 16, 32

        # Velocity
        v_angle = (np.arctan2(vy, vx) + 2 * np.pi) % (2 * np.pi)
        v_angle_bin = int(v_angle / (2 * np.pi) * self.num_angle_bins)
        v_norm = np.sqrt(vx**2 + vy**2)
        v_norm_bin = np.digitize([v_norm], self.velocity_bins)[0]

        # Wind
        w_angle = (np.arctan2(wy, wx) + 2 * np.pi) % (2 * np.pi)
        w_angle_bin = int(w_angle / (2 * np.pi) * self.num_angle_bins)
        w_norm = np.sqrt(wx**2 + wy**2)
        w_norm_bin = np.digitize([w_norm], self.wind_bins)[0]

        # Goal vector
        gx, gy = goal_x - x, goal_y - y
        g_angle = (np.arctan2(gy, gx) + 2 * np.pi) % (2 * np.pi)
        g_angle_bin = int(g_angle / (2 * np.pi) * self.num_angle_bins)
        g_norm = np.sqrt(gx**2 + gy**2)
        g_norm_bin = np.digitize([g_norm], self.goal_dist_bins)[0]

        return (v_angle_bin, v_norm_bin, w_angle_bin, w_norm_bin, g_angle_bin, g_norm_bin)
        
    def act(self, observation):
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
    
    def learn(self, state, action, reward, next_state, next_action=None):
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
    
    def decay(self):
        pass

    def reset(self):
        """Reset the agent for a new episode."""
        # Nothing to reset for Q-learning agent
        pass
        
    def seed(self, seed=None):
        """Set the random seed."""
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
