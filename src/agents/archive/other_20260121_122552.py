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
        # Reduced state space for faster learning
        self.num_angle_bins = 16
        # Simplified speed/dist bins
        self.velocity_bins = [0.5, 1.5] # 3 states: <0.5 (stopped/slow), 0.5-1.5 (moderate), >1.5 (fast)
        self.goal_dist_bins = [5, 15, 30] # 4 states: very close, close, medium, far
        
        # Initialize Q-table
        # New State space: 
        # 1. Angle(Wind, BoatVelocity) - Index relative to wind
        # 2. Angle(Goal, BoatVelocity) - Direction to goal relative to movement
        # 3. Boat Speed
        # 4. Goal Distance
        self.q_table = {}

        
    def discretize_state(self, observation):
        """Convert continuous observation to discrete state using relative coordinates."""
        # Extract raw observations
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        goal_x, goal_y = 16, 32 # Assuming fixed goal for consistency with original code

        # 1. Calculate base angles (0-2pi)
        # Velocity angle (Direction of movement)
        if abs(vx) < 0.1 and abs(vy) < 0.1:
            # If stopped, use wind direction as reference or keep previous? 
            # For simplicity, treat as North (0,1) or use last known.
            # Using 0 (East in atan2) is risky. Let's assume North-East default.
            v_angle = np.pi / 2 
        else:
            v_angle = np.arctan2(vy, vx)
            
        # Wind angle
        w_angle = np.arctan2(wy, wx)
        
        # Goal vector and angle
        gx, gy = goal_x - x, goal_y - y
        g_angle = np.arctan2(gy, gx)
        
        # 2. Relative Angles (Normalize to 0-2pi)
        
        # Angle of Wind relative to Boat Velocity (Where is the wind coming from relative to me?)
        # 0 = Wind from behind me (Tailwind), Pi = Headwind
        rel_wind_angle = (w_angle - v_angle + 2 * np.pi) % (2 * np.pi)
        rel_wind_bin = int(rel_wind_angle / (2 * np.pi) * self.num_angle_bins)
        
        # Angle of Goal relative to Boat Velocity (Where is the goal relative to where I'm going?)
        # 0 = I'm heading straight for the goal
        rel_goal_angle = (g_angle - v_angle + 2 * np.pi) % (2 * np.pi)
        rel_goal_bin = int(rel_goal_angle / (2 * np.pi) * self.num_angle_bins)

        # 3. Magnitudes
        v_norm = np.sqrt(vx**2 + vy**2)
        v_bin = np.digitize([v_norm], self.velocity_bins)[0]

        g_norm = np.sqrt(gx**2 + gy**2)
        g_bin = np.digitize([g_norm], self.goal_dist_bins)[0]

        # State tuple: (Relative Wind, Relative Goal, Speed, Distance)
        # Size: 16 * 16 * 3 * 4 = 3072 states
        return (rel_wind_bin, rel_goal_bin, v_bin, g_bin)
        
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
