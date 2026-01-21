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

        # Learning rate
        self.INITIAL_LEARNING_RATE = 0.1
        self.learning_rate = self.INITIAL_LEARNING_RATE
        self.learning_rate_decay = 0.999
        self.learning_rate_min = 0.01

        # Exploration rate
        self.INITIAL_EXPLORATION_RATE = 0.5
        self.exploration_rate = self.INITIAL_EXPLORATION_RATE
        self.exploration_rate_decay = 0.995
        self.exploration_rate_min = 0.01

        # Discount factor
        self.discount_factor = 0.9

        
        # Discretization parameters
        self.num_angle_bins = 16
        self.velocity_bins = [0.2, 0.5, 1, 2, 5]
        self.wind_bins = [0.5, 1, 2, 5]
        self.goal_dist_bins = [5, 10, 20, 30, 45]
        
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
        # Calculer l'angle du vent
        wx, wy = observation[4], observation[5]
        wind_angle = np.arctan2(wy, wx)
        
        # Filtrer les actions dans la no-go zone (±45° du vent)
        valid_actions = []
        for action in range(9):
            action_angle = action * (2 * np.pi / 8)  # Approximatif
            angle_diff = abs(action_angle - wind_angle)
            if angle_diff > np.pi/4:  # Plus de 45° du vent
                valid_actions.append(action)
        
        # Explorer parmi les actions valides seulement
        if not valid_actions:
            valid_actions = list(range(9))  # Fallback
        
        if self.np_random.random() < self.exploration_rate:
            return self.np_random.choice(valid_actions)
        else:
            # Exploiter parmi les actions valides
            state = self.discretize_state(observation)
            if state not in self.q_table:
                self.q_table[state] = np.zeros(9)
            
            q_values = self.q_table[state].copy()
            invalid_mask = [i for i in range(9) if i not in valid_actions]
            q_values[invalid_mask] = -np.inf
            return np.argmax(q_values)
        
    def learn(self, state, action, reward, next_state, next_action):
        """Update Q-table based on observed transition."""
        # Initialize Q-values if states not in table
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)

        reward -= 5 # time penalty
        
        # Q-learning update
        td_target = reward + self.discount_factor * np.max(self.q_table[next_state])
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
    
    def decay(self):
        self.learning_rate = max(self.learning_rate * self.learning_rate_decay, self.learning_rate_min)
        self.exploration_rate = max(self.exploration_rate * self.exploration_rate_decay, self.exploration_rate_min)

    def reset(self):
        """Reset the agent for a new episode."""
        # Nothing to reset for Q-learning agent
        self.learning_rate = self.INITIAL_LEARNING_RATE
        self.exploration_rate = self.INITIAL_EXPLORATION_RATE
        
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
