"""
Agent with VMG (Velocity Made Good) - Projected velocity towards goal.
Based on physic_wind_with_optimal_angle.py with added VMG reward shaping.
"""

import numpy as np
from agents.base_agent import BaseAgent
from src.sailing_physics import calculate_sailing_efficiency


class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

        # Learning parameters with decay
        self.learning_rate = 0.2         # alpha - start high
        self.min_learning_rate = 0.05    # minimum alpha
        self.lr_decay_rate = 0.999       # decay per episode
        
        self.discount_factor = 0.99  # gamma
        
        # Epsilon with decay (per episode, not per step)
        self.exploration_rate = 0.3      # Start high for exploration
        self.min_exploration = 0.01      # Minimum epsilon
        self.eps_decay_rate = 0.995      # Decay per episode (slower)

        # Discretization parameters
        self.position_bins = 8
        self.velocity_bins = 5
        self.wind_bins = 8
        self.wind_preview_steps = 3
        self.grid_size = 32
        
        # Goal position (fixed at top center)
        self.goal_position = np.array([16, 31])
        
        # Q-table
        self.q_table = {}
        self.q_init_high = 10.0
        
        # State for learning
        self.last_efficiency = 0.0
        self.last_vmg = 0.0

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

        return (x_bin, y_bin, v_bin, wind_bin)

    def _calculate_vmg(self, observation, action):
        """
        VMG intelligent pondéré par l'efficacité de navigation.
        
        Combine:
        - La direction vers le goal (cos_to_goal)
        - L'efficacité de navigation avec le vent local (polar diagram)
        
        Returns a value between 0 and 1.
        """
        # Si action = 8 (stay), VMG = 0
        if action == 8:
            return 0.0
        
        # Extraire position et vent de l'observation
        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5]
        
        position = np.array([x, y])
        wind_vec = np.array([wx, wy])
        wind_mag = np.linalg.norm(wind_vec)
        
        # Direction vers le goal
        direction_to_goal = self.goal_position - position
        dist_to_goal = np.linalg.norm(direction_to_goal)
        
        if dist_to_goal < 0.1:  # Déjà au goal
            return 1.0
            
        direction_to_goal = direction_to_goal / dist_to_goal
        
        # Direction du bateau (action choisie)
        action_vec = self._action_to_direction(action)
        action_norm = np.linalg.norm(action_vec)
        if action_norm > 0:
            action_vec = action_vec / action_norm
        
        # Efficacité de cette direction avec le vent local
        if wind_mag > 0:
            wind_normalized = wind_vec / wind_mag
            efficiency = calculate_sailing_efficiency(action_vec, wind_normalized)
        else:
            efficiency = 1.0  # Pas de vent = efficacité max
        
        # Composante vers le goal (entre -1 et 1)
        cos_to_goal = np.dot(action_vec, direction_to_goal)
        
        # VMG "intelligent" = efficacité × max(0, cos_to_goal)
        # On garde seulement les directions qui vont vers le goal
        smart_vmg = efficiency * max(0, cos_to_goal)
        
        return smart_vmg

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

        # Calculate VMG for reward shaping
        self.last_vmg = self._calculate_vmg(observation, action)

        return action

    def learn(self, state, action, reward, next_state, next_action=None):
        """SARSA learning: uses next_action for Q(s',a') instead of max(Q(s'))"""
        if state not in self.q_table:
            self.q_table[state] = self.np_random.random(9) * self.q_init_high
        if next_state not in self.q_table:
            self.q_table[next_state] = self.np_random.random(9) * self.q_init_high

        # Efficiency bonus (polar diagram) - reduced to let VMG dominate
        efficiency_bonus = self.last_efficiency * 0.3
        
        # VMG bonus - strongly reward moving towards the goal
        vmg_bonus = self.last_vmg * 10.0
        
        # Step penalty - discourage long trajectories
        step_penalty = 0.3
        
        # Combined shaped reward
        shaped_reward = reward + efficiency_bonus + vmg_bonus - step_penalty
        
        # SARSA update: Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
        td_target = shaped_reward + self.discount_factor * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.learning_rate * td_error

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_vmg = 0.0
        
        # Decay epsilon and learning rate (per episode)
        self.exploration_rate = max(self.min_exploration, 
                                    self.exploration_rate * self.eps_decay_rate)
        self.learning_rate = max(self.min_learning_rate,
                                 self.learning_rate * self.lr_decay_rate)

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
