"""
Agent with improved VMG (Velocity Made Good) - Fixed version based on technical critique.

Key improvements:
1. VMG based on actual movement (delta_pos), not action intention
2. Proper zero/low wind handling (efficiency -> 0, not 1.0)
3. Turn/maneuver costs to prevent free zigzagging
4. Viable "stay" action in low wind conditions
5. Better reward balancing
6. Enhanced state representation with direction memory
"""

import numpy as np
from agents.base_agent import BaseAgent
from src.sailing_physics import calculate_sailing_efficiency


class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

        # Learning parameters with decay
        self.learning_rate = 0.15         # alpha - start high
        self.min_learning_rate = 0.05    # minimum alpha
        self.lr_decay_rate = 0.999       # decay per episode
        
        self.discount_factor = 0.99  # gamma
        
        # Epsilon with decay (per episode, not per step)
        self.exploration_rate = 0.15      # Start high for exploration
        self.min_exploration = 0.01      # Minimum epsilon
        self.eps_decay_rate = 0.995      # Decay per episode (slower)

        # Discretization parameters
        self.position_bins = 10
        self.velocity_bins = 5
        self.wind_bins = 8
        self.wind_preview_steps = 3
        self.grid_size = 32
        
        # Goal position (fixed at top center)
        self.goal_position = np.array([16, 31])
        
        # Q-table
        self.q_table = {}
        self.q_init_high = 5.0  # Reduced from 10.0 to avoid noise
        
        # State tracking for learning
        self.last_position = None
        self.last_action = None
        self.last_efficiency = 0.0
        self.last_vmg = 0.0
        
        # Wind thresholds
        self.low_wind_threshold = 0.3  # Below this, special handling kicks in
        self.epsilon = 1e-6  # Numerical stability

    def _action_to_direction(self, action):
        """Convert action index to direction vector."""
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1),
            (0, 0)
        ]
        return np.array(directions[action])

    def discretize_state(self, observation):
        """
        Discretize the continuous observation into a tuple state.
        Enhanced with last action to provide direction memory.
        """
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

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

        # Include last action for direction memory (helps prevent oscillations)
        last_action_bin = self.last_action if self.last_action is not None else 8
        
        return (x_bin, y_bin, v_bin, wind_bin, last_action_bin)

    def _calculate_vmg_from_movement(self, old_position, new_position):
        """
        Calculate VMG based on ACTUAL movement, not action intention.
        
        This is the key fix: VMG should reflect real displacement towards goal.
        """
        # Actual displacement
        delta_pos = new_position - old_position
        
        # Direction to goal from new position
        direction_to_goal = self.goal_position - new_position
        dist_to_goal = np.linalg.norm(direction_to_goal)
        
        if dist_to_goal < self.epsilon:
            return 1.0  # At goal
        
        direction_to_goal = direction_to_goal / dist_to_goal
        
        # VMG = projection of movement onto goal direction
        vmg = np.dot(delta_pos, direction_to_goal)
        
        # Normalize by a reasonable max speed (adjust based on your environment)
        max_possible_speed = 2.0  # Tune this based on your environment
        vmg_normalized = vmg / max_possible_speed
        
        # Clamp to [0, 1] for reward shaping
        return np.clip(vmg_normalized, 0.0, 1.0)

    def _calculate_sailing_efficiency_safe(self, action_vec, wind_vec, wind_mag):
        """
        Calculate sailing efficiency with proper zero-wind handling.
        
        Key fix: efficiency -> 0 when wind -> 0, not 1.0
        """
        if wind_mag < self.epsilon:
            return 0.0  # No wind = no propulsion
        
        wind_normalized = wind_vec / wind_mag
        base_efficiency = calculate_sailing_efficiency(action_vec, wind_normalized)
        
        # Scale efficiency by wind magnitude (low wind = low efficiency)
        # This ensures VMG -> 0 as wind -> 0
        wind_factor = min(wind_mag / 2.0, 1.0)  # Normalize to typical wind strength
        
        return base_efficiency * wind_factor

    def act(self, observation, info=None):
        """
        Select an action using epsilon-greedy strategy combined with physical sailing rules.
        """
        # Store current position for VMG calculation
        current_position = np.array([observation[0], observation[1]])
        
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
            
            if wind_mag > self.epsilon:
                cos_theta = (ax * wx + ay * wy) / (a_mag * wind_mag)
                
                if cos_theta >= no_go_threshold:
                    valid_actions.append(i)
            else:
                # No wind: all directions valid (but will get low efficiency)
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

        # Calculate efficiency of the chosen action (for reward shaping)
        if action < 8:
            action_vec = self._action_to_direction(action)
            action_vec = action_vec / np.linalg.norm(action_vec)
            
            wind_vec = np.array([wx, wy])
            self.last_efficiency = self._calculate_sailing_efficiency_safe(
                action_vec, wind_vec, wind_mag
            )
        else:
            self.last_efficiency = 0.0

        # Calculate VMG from actual movement (if we have previous position)
        if self.last_position is not None:
            self.last_vmg = self._calculate_vmg_from_movement(
                self.last_position, current_position
            )
        else:
            self.last_vmg = 0.0

        # Update tracking
        self.last_position = current_position.copy()
        self.last_action = action

        return action

    def learn(self, state, action, reward, next_state, next_action=None):
        """
        SARSA learning with improved reward shaping.
        
        Key improvements:
        1. Reduced VMG weight (was 10.0, now 2.0)
        2. Turn penalty to discourage zigzagging
        3. Viable "stay" action in low wind
        4. Better balance between components
        """
        if state not in self.q_table:
            self.q_table[state] = self.np_random.random(9) * self.q_init_high
        if next_state not in self.q_table:
            self.q_table[next_state] = self.np_random.random(9) * self.q_init_high

        # Extract last action from state (state includes last_action now)
        prev_action = state[4]  # last_action_bin is 5th element
        
        # --- Reward shaping components ---
        
        # 1. Efficiency bonus (polar diagram) - keep moderate
        efficiency_bonus = self.last_efficiency * 0.5
        
        # 2. VMG bonus - REDUCED from 10.0 to 2.0 (key fix!)
        vmg_bonus = self.last_vmg * 2.0
        
        # 3. Step penalty - INCREASED to discourage long trajectories
        step_penalty = 1
        
        # 4. Turn penalty - NEW! Discourage zigzagging
        turn_penalty = 0.0
        if prev_action != 8 and action != 8 and prev_action != action:
            # Calculate angular difference between actions
            prev_dir = self._action_to_direction(prev_action)
            curr_dir = self._action_to_direction(action)
            
            # Penalize based on turn magnitude
            if np.linalg.norm(prev_dir) > 0 and np.linalg.norm(curr_dir) > 0:
                prev_dir = prev_dir / np.linalg.norm(prev_dir)
                curr_dir = curr_dir / np.linalg.norm(curr_dir)
                
                # Dot product: 1 = same direction, -1 = opposite
                alignment = np.dot(prev_dir, curr_dir)
                
                # Penalty increases as alignment decreases
                # 0 penalty for same direction, 0.3 for opposite
                turn_penalty = 0.3 * (1.0 - alignment) / 2.0
        
        # 5. Stay action handling - make it viable in low wind
        stay_bonus = 0.0
        if action == 8:
            # Check wind from observation (need to extract from somewhere)
            # For now, use efficiency as proxy: low efficiency -> likely low wind
            if self.last_efficiency < 0.2:
                stay_bonus = 0.2  # Small bonus for waiting in low wind
        
        # Combined shaped reward
        shaped_reward = (
            reward 
            + efficiency_bonus 
            + vmg_bonus 
            - step_penalty 
            - turn_penalty 
            + stay_bonus
        )
        
        # SARSA update: Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
        td_target = shaped_reward + self.discount_factor * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.learning_rate * td_error

    def reset(self):
        """Reset episode-level state."""
        self.last_position = None
        self.last_action = None
        self.last_vmg = 0.0
        self.last_efficiency = 0.0
        
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
