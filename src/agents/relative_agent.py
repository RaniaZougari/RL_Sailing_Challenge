
import numpy as np
from agents.base_agent import BaseAgent
from src.sailing_physics import calculate_sailing_efficiency

class MyAgent(BaseAgent):
    """
    Relative State Agent:
    Uses a state representation based on relative angles (Wind-Goal, Heading-Goal)
    to generalize sailing strategies (e.g. tacking) across different positions.
    """
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

        # Learning parameters with decay
        self.learning_rate = 0.2         # alpha - start high
        self.min_learning_rate = 0.05    # minimum alpha
        self.lr_decay_rate = 0.999       # decay per episode
        
        self.discount_factor = 0.99  # gamma
        
        # Epsilon with decay (per episode)
        self.exploration_rate = 0.3      # Start high
        self.min_exploration = 0.01      # Minimum epsilon
        self.eps_decay_rate = 0.995      # Decay per episode
        
        # State Discretization parameters
        # Relative Wind Angle (Goal frame): 0-7 (45 deg bins)
        # Relative Heading (Wind frame? or Goal frame?): 0-7
        # Speed: 0-2 (Stop, Slow, Fast)
        self.angle_bins = 8
        self.speed_bins = 3
        
        # Goal position (fixed at top center for this challenge)
        self.goal_position = np.array([16, 31])
        
        # Q-table
        self.q_table = {}
        self.q_init_high = 10.0
        
        # State for learning
        self.last_efficiency = 0.0
        self.last_vmg = 0.0
        self.prev_vmg = 0.0 # VMG of the action taken in the PREVIOUS step
        
        # Physics Mask Threshold
        # Allow efficiency >= 0.4 to permit 45-degree tacks (eff=0.5)
        self.mask_threshold = 0.4

    def _action_to_direction(self, action):
        """Convert action index to direction vector."""
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1),
            (0, 0) # Action 8: Hold
        ]
        if action < 8:
            return np.array(directions[action], dtype=float)
        return np.array([0, 0], dtype=float)
        
    def discretize_state(self, observation):
        """
        Convert continuous observation to a discrete state tuple using RELATIVE coordinates.
        State: (wind_relative_to_goal, heading_relative_to_goal, speed_bin)
        """
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        
        # 1. Calculate key vectors/angles
        # Vector/Angle to Goal
        to_goal = self.goal_position - np.array([x, y])
        dist_goal = np.linalg.norm(to_goal)
        if dist_goal < 0.001:
            angle_to_goal = 0.0
        else:
            angle_to_goal = np.arctan2(to_goal[1], to_goal[0])
            
        # Wind vector angle
        angle_wind = np.arctan2(wy, wx)
        
        # Velocity vector angle (Heading)
        speed = np.linalg.norm([vx, vy])
        if speed < 0.001:
            angle_heading = angle_to_goal
        else:
            angle_heading = np.arctan2(vy, vx)
            
        # 2. Compute Relative Angles (normalized to [-pi, pi])
        
        # Wind Direction relative to Goal Direction
        rel_wind = angle_wind - angle_to_goal
        rel_wind = (rel_wind + np.pi) % (2 * np.pi) - np.pi
        
        # Heading relative to Goal Direction
        rel_heading = angle_heading - angle_to_goal
        rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi
        
        # 3. Discretize
        def get_bin(angle, num_bins):
            # Normalize to [0, 2pi)
            angle = (angle + 2*np.pi) % (2*np.pi)
            # Bin 0 is centered at 0 (from -pi/num_bins to pi/num_bins)
            bin_idx = int((angle + np.pi/num_bins) / (2*np.pi) * num_bins) % num_bins
            return bin_idx
            
        wind_bin = get_bin(rel_wind, self.angle_bins)
        heading_bin = get_bin(rel_heading, self.angle_bins)
        
        # Speed bin
        if speed < 0.1:
            speed_bin = 0 # Stopped
        elif speed < 1.0:
            speed_bin = 1 # Slow/Tacking
        else:
            speed_bin = 2 # Fast
            
        return (wind_bin, heading_bin, speed_bin)

    def _calculate_vmg_reward(self, position, action, wind_vec):
        """Calculate VMG-based reward component."""
        direction_to_goal = self.goal_position - position
        dist_to_goal = np.linalg.norm(direction_to_goal)
        
        if dist_to_goal < 0.1: return 1.0
        
        direction_to_goal /= dist_to_goal
        action_vec = self._action_to_direction(action)
        action_norm = np.linalg.norm(action_vec)
        if action_norm > 0: action_vec /= action_norm
        
        wind_mag = np.linalg.norm(wind_vec)
        if wind_mag > 0:
            efficiency = calculate_sailing_efficiency(action_vec, wind_vec/wind_mag)
        else:
            efficiency = 1.0
            
        cos_to_goal = np.dot(action_vec, direction_to_goal)
        
        # VMG = Efficiency * cos(theta_to_goal)
        vmg = efficiency * cos_to_goal
        return vmg

    def act(self, observation, info=None):
        state = self.discretize_state(observation)
        
        if state not in self.q_table:
            self.q_table[state] = self.np_random.random(9) * self.q_init_high
            
        # Physics filtering
        wx, wy = observation[4], observation[5]
        wind_mag = np.sqrt(wx**2 + wy**2)
        
        valid_actions = []
        action_vectors = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        
        # Check efficiency for all actions 0-7
        if wind_mag > 0:
            wind_vec = np.array([wx, wy]) / wind_mag
            for i, vec in enumerate(action_vectors):
                vec_np = np.array(vec) / np.linalg.norm(vec)
                eff = calculate_sailing_efficiency(vec_np, wind_vec)
                
                # LOWERED THRESHOLD to 0.4 to allow 45-degree tacks (eff=0.5)
                if eff >= self.mask_threshold: 
                    valid_actions.append(i)
        
        if not valid_actions: valid_actions = [0,1,2,3,4,5,6,7]
        
        # Epsilon-Greedy
        if self.np_random.random() < self.exploration_rate:
            action_idx = self.np_random.choice(valid_actions)
        else:
            # Pick best valid action
            q_values = self.q_table[state]
            masked_q = np.full(9, -np.inf)
            masked_q[valid_actions] = q_values[valid_actions]
            action_idx = int(np.argmax(masked_q))
            
        # Calculate VMG for the chosen action for reward shaping
        x, y = observation[0], observation[1]
        position = np.array([x, y])
        
        # Update VMG tracking for learning
        # prev_vmg holds the VMG of the action from the previous step (s, a)
        # last_vmg holds the VMG of the action from the current step (s', a')
        self.prev_vmg = self.last_vmg
        
        current_vmg = self._calculate_vmg_reward(position, action_idx, np.array([wx, wy]))
        self.last_vmg = current_vmg
        
        return int(action_idx)

    def learn(self, state, action, reward, next_state, next_action=None):
        if state not in self.q_table:
            self.q_table[state] = self.np_random.random(9) * self.q_init_high
        if next_state not in self.q_table:
            self.q_table[next_state] = self.np_random.random(9) * self.q_init_high
            
        # Reward Shaping: REWARD + VMG_BONUS - PENALTY
        # Use simple shaped reward: Reward + VMG * Multiplier
        # self.prev_vmg corresponds to the action `action` taken at `state`.
        vmg_bonus = self.prev_vmg * 15.0
        step_penalty = 0.5
        
        shaped_reward = reward + vmg_bonus - step_penalty
        
        if next_action is not None:
             td_target = shaped_reward + self.discount_factor * self.q_table[next_state][next_action]
        else:
             td_target = shaped_reward + self.discount_factor * np.nanmax(self.q_table[next_state])
             
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        
    def reset(self):
        self.last_vmg = 0.0
        self.prev_vmg = 0.0
        
        # Decay
        self.exploration_rate = max(self.min_exploration, 
                                    self.exploration_rate * self.eps_decay_rate)
        self.learning_rate = max(self.min_learning_rate,
                                 self.learning_rate * self.lr_decay_rate)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                'q_table': self.q_table,
                'exploration_rate': self.exploration_rate,
                'learning_rate': self.learning_rate
            }, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.exploration_rate = data.get('exploration_rate', 0.01)
            self.learning_rate = data.get('learning_rate', 0.05)
