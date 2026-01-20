"""
Agent with CORRECTLY implemented VMG (Velocity Made Good) - v3.

Critical fixes from ChatGPT critique:
1. VMG calculated in learn() AFTER movement, not in act() BEFORE
2. Proper position tracking (current_position stored in act())
3. Wind magnitude stored for stay bonus calculation
4. Turn penalty uses correct prev_action from state
5. All timing issues resolved
"""

import numpy as np
from agents.base_agent import BaseAgent
from src.sailing_physics import calculate_sailing_efficiency


class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

        # Learning parameters with decay
        self.learning_rate = 0.15
        self.min_learning_rate = 0.05
        self.lr_decay_rate = 0.999
        
        self.discount_factor = 0.99
        
        # Epsilon with decay
        self.exploration_rate = 0.09
        self.min_exploration = 0.01
        self.eps_decay_rate = 0.995

        # Discretization parameters
        self.position_bins = 10
        self.velocity_bins = 5
        self.wind_bins = 8
        self.grid_size = 32
        
        # Goal position
        self.goal_position = np.array([16.0, 31.0])
        
        # Q-table
        self.q_table = {}
        self.q_init_high = 5.0  # Reduced from 10.0
        
        # State tracking - CRITICAL for VMG calculation
        self.last_position = None      # Position at previous step
        self.current_position = None   # Position at current step (stored in act())
        self.last_action = None        # Action taken at previous step
        self.last_wind_mag = 0.0       # Wind magnitude at current step
        self.last_efficiency = 0.0     # Efficiency of current action
        self.last_vmg = 0.0            # VMG calculated in learn()
        
        # Thresholds
        self.low_wind_threshold = 0.3
        self.epsilon = 1e-6

    def _action_to_direction(self, action):
        """Convert action index to direction vector."""
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1),
            (0, 0)
        ]
        return np.array(directions[action], dtype=np.float64)

    def discretize_state(self, observation):
        """
        Discretize observation with last action for direction memory.
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

        # Include last action for direction memory
        last_action_bin = self.last_action if self.last_action is not None else 8
        
        return (x_bin, y_bin, v_bin, wind_bin, last_action_bin)

    def _calculate_vmg_from_movement(self, old_position, new_position):
        """
        Calculate VMG based on ACTUAL movement.
        
        This is called in learn() AFTER the environment step.
        """
        # Actual displacement
        delta_pos = new_position - old_position
        displacement = np.linalg.norm(delta_pos)
        
        # Direction to goal from new position
        direction_to_goal = self.goal_position - new_position
        dist_to_goal = np.linalg.norm(direction_to_goal)
        
        if dist_to_goal < self.epsilon:
            return 1.0  # At goal
        
        if displacement < 0.01:
            return 0.0  # Almost no movement
        
        direction_to_goal = direction_to_goal / dist_to_goal
        
        # VMG = projection of movement onto goal direction
        vmg = np.dot(delta_pos, direction_to_goal)
        
        # Normalize by max possible speed
        max_possible_speed = 2.0
        vmg_normalized = vmg / max_possible_speed
        
        return np.clip(vmg_normalized, 0.0, 1.0)

    def _calculate_sailing_efficiency_safe(self, action_vec, wind_vec, wind_mag):
        """
        Calculate sailing efficiency with proper zero-wind handling.
        """
        if wind_mag < self.epsilon:
            return 0.0  # No wind = no propulsion
        
        wind_normalized = wind_vec / wind_mag
        base_efficiency = calculate_sailing_efficiency(action_vec, wind_normalized)
        
        # Scale by wind magnitude
        wind_factor = min(wind_mag / 2.0, 1.0)
        
        return base_efficiency * wind_factor

    def act(self, observation, info=None):
        """
        Select action using epsilon-greedy with physics constraints.
        
        CRITICAL: Store current_position and wind_mag for use in learn().
        """
        # Store current position (will be used as "new_position" in learn())
        self.current_position = np.array([observation[0], observation[1]], dtype=np.float64)
        
        # Store wind magnitude for stay bonus calculation
        wx, wy = observation[4], observation[5]
        self.last_wind_mag = np.sqrt(wx**2 + wy**2)
        
        state = self.discretize_state(observation)

        if state not in self.q_table:
            self.q_table[state] = self.np_random.random(9) * self.q_init_high

        # Physics-based action filtering
        valid_actions = []
        action_vectors = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        
        no_go_threshold = -0.707

        for i in range(8):
            ax, ay = action_vectors[i]
            a_mag = np.sqrt(ax**2 + ay**2)
            
            if self.last_wind_mag > self.epsilon:
                cos_theta = (ax * wx + ay * wy) / (a_mag * self.last_wind_mag)
                if cos_theta >= no_go_threshold:
                    valid_actions.append(i)
            else:
                valid_actions.append(i)
                
        valid_actions.append(8)  # Stay always valid

        # Epsilon-greedy
        if self.np_random.random() < self.exploration_rate:
            action = self.np_random.choice(valid_actions)
        else:
            q_values = self.q_table[state].copy()
            mask = np.full(9, -np.inf)
            mask[valid_actions] = 0
            masked_q_values = q_values + mask
            action = np.nanargmax(masked_q_values)

        # Calculate efficiency for reward shaping
        if action < 8:
            action_vec = self._action_to_direction(action)
            action_vec = action_vec / np.linalg.norm(action_vec)
            
            wind_vec = np.array([wx, wy], dtype=np.float64)
            self.last_efficiency = self._calculate_sailing_efficiency_safe(
                action_vec, wind_vec, self.last_wind_mag
            )
        else:
            self.last_efficiency = 0.0

        return action

    def learn(self, state, action, reward, next_state, next_action=None):
        """
        SARSA learning with corrected reward shaping.
        
        CRITICAL: VMG is calculated HERE, after the environment step.
        """
        if state not in self.q_table:
            self.q_table[state] = self.np_random.random(9) * self.q_init_high
        if next_state not in self.q_table:
            self.q_table[next_state] = self.np_random.random(9) * self.q_init_high

        # Calculate VMG from ACTUAL movement (now we have both positions)
        if self.last_position is not None and self.current_position is not None:
            self.last_vmg = self._calculate_vmg_from_movement(
                self.last_position,
                self.current_position
            )
        else:
            self.last_vmg = 0.0

        # Extract previous action from state (for turn penalty)
        prev_action = state[4]  # last_action_bin is 5th element
        
        # --- Reward shaping components ---
        
        # 1. Efficiency bonus
        efficiency_bonus = self.last_efficiency * 0.5
        
        # 2. VMG bonus - REDUCED from 10.0 to 2.0
        vmg_bonus = self.last_vmg * 2.0
        
        # 3. Step penalty
        step_penalty = 0.5
        
        # 4. Turn penalty - penalize direction changes
        turn_penalty = 0.0
        if prev_action != 8 and action != 8 and prev_action != action:
            prev_dir = self._action_to_direction(prev_action)
            curr_dir = self._action_to_direction(action)
            
            if np.linalg.norm(prev_dir) > 0 and np.linalg.norm(curr_dir) > 0:
                prev_dir = prev_dir / np.linalg.norm(prev_dir)
                curr_dir = curr_dir / np.linalg.norm(curr_dir)
                
                alignment = np.dot(prev_dir, curr_dir)
                
                # Penalty: 0 for same direction, 0.3 for opposite
                turn_penalty = 0.3 * (1.0 - alignment) / 2.0
        
        # 5. Stay bonus in low wind
        stay_bonus = 0.0
        if action == 8 and self.last_wind_mag < self.low_wind_threshold:
            stay_bonus = 0.3  # Compensates step_penalty
        
        # Combined shaped reward
        shaped_reward = (
            reward 
            + efficiency_bonus 
            + vmg_bonus 
            - step_penalty 
            - turn_penalty 
            + stay_bonus
        )
        
        # SARSA update
        td_target = shaped_reward + self.discount_factor * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

        # Update position tracking for next step
        # CRITICAL: This happens AFTER VMG calculation
        self.last_position = self.current_position.copy() if self.current_position is not None else None
        self.last_action = action

    def reset(self):
        """Reset episode-level state."""
        self.last_position = None
        self.current_position = None
        self.last_action = None
        self.last_vmg = 0.0
        self.last_efficiency = 0.0
        self.last_wind_mag = 0.0
        
        # Decay epsilon and learning rate
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
