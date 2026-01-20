"""
Agent with Expected SARSA - v4.

Based on vmg_agent_v2 but using Expected SARSA instead of standard SARSA.

Key difference:
- Standard SARSA: Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
- Expected SARSA: Q(s,a) ← Q(s,a) + α[R + γ·E[Q(s',·)] - Q(s,a)]
  where E[Q(s',·)] = Σ π(a'|s') Q(s',a')

Expected SARSA is more stable than SARSA because it averages over all possible
next actions weighted by their probability under the current policy.
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
        self.wind_preview_steps = 3
        self.grid_size = 32
        
        # Goal position
        self.goal_position = np.array([16, 31])
        
        # Q-table
        self.q_table = {}
        self.q_init_high = 5.0
        
        # State tracking
        self.last_position = None
        self.last_action = None
        self.last_efficiency = 0.0
        self.last_vmg = 0.0
        
        # Wind thresholds
        self.low_wind_threshold = 0.3
        self.epsilon = 1e-6

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
        """
        delta_pos = new_position - old_position
        
        direction_to_goal = self.goal_position - new_position
        dist_to_goal = np.linalg.norm(direction_to_goal)
        
        if dist_to_goal < self.epsilon:
            return 1.0
        
        direction_to_goal = direction_to_goal / dist_to_goal
        
        vmg = np.dot(delta_pos, direction_to_goal)
        
        max_possible_speed = 2.0
        vmg_normalized = vmg / max_possible_speed
        
        return np.clip(vmg_normalized, 0.0, 1.0)

    def _calculate_sailing_efficiency_safe(self, action_vec, wind_vec, wind_mag):
        """
        Calculate sailing efficiency with proper zero-wind handling.
        """
        if wind_mag < self.epsilon:
            return 0.0
        
        wind_normalized = wind_vec / wind_mag
        base_efficiency = calculate_sailing_efficiency(action_vec, wind_normalized)
        
        wind_factor = min(wind_mag / 2.0, 1.0)
        
        return base_efficiency * wind_factor

    def _get_valid_actions(self, observation):
        """
        Get valid actions based on physics constraints.
        Returns list of valid action indices.
        """
        wx, wy = observation[4], observation[5]
        wind_mag = np.sqrt(wx**2 + wy**2)
        
        valid_actions = []
        action_vectors = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        
        no_go_threshold = -0.707

        for i in range(8):
            ax, ay = action_vectors[i]
            a_mag = np.sqrt(ax**2 + ay**2)
            
            if wind_mag > self.epsilon:
                cos_theta = (ax * wx + ay * wy) / (a_mag * wind_mag)
                if cos_theta >= no_go_threshold:
                    valid_actions.append(i)
            else:
                valid_actions.append(i)
                
        valid_actions.append(8)  # Stay always valid
        return valid_actions

    def act(self, observation, info=None):
        """
        Select action using epsilon-greedy with physics constraints.
        """
        current_position = np.array([observation[0], observation[1]])
        
        state = self.discretize_state(observation)

        if state not in self.q_table:
            self.q_table[state] = self.np_random.random(9) * self.q_init_high

        # Get valid actions
        wx, wy = observation[4], observation[5]
        wind_mag = np.sqrt(wx**2 + wy**2)
        valid_actions = self._get_valid_actions(observation)

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
            
            wind_vec = np.array([wx, wy])
            self.last_efficiency = self._calculate_sailing_efficiency_safe(
                action_vec, wind_vec, wind_mag
            )
        else:
            self.last_efficiency = 0.0

        # Calculate VMG from actual movement
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
        Expected SARSA learning with improved reward shaping.
        
        Key difference from standard SARSA:
        Instead of using Q(s', a'), we use the expected value over all actions
        in the next state, weighted by the policy probabilities.
        """
        if state not in self.q_table:
            self.q_table[state] = self.np_random.random(9) * self.q_init_high
        if next_state not in self.q_table:
            self.q_table[next_state] = self.np_random.random(9) * self.q_init_high

        # Extract previous action from state
        prev_action = state[4]
        
        # --- Reward shaping components ---
        
        # 1. Efficiency bonus
        efficiency_bonus = self.last_efficiency * 0.5
        
        # 2. VMG bonus
        vmg_bonus = self.last_vmg * 2.0
        
        # 3. Step penalty
        step_penalty = 1.0
        
        # 4. Turn penalty
        turn_penalty = 0.0
        if prev_action != 8 and action != 8 and prev_action != action:
            prev_dir = self._action_to_direction(prev_action)
            curr_dir = self._action_to_direction(action)
            
            if np.linalg.norm(prev_dir) > 0 and np.linalg.norm(curr_dir) > 0:
                prev_dir = prev_dir / np.linalg.norm(prev_dir)
                curr_dir = curr_dir / np.linalg.norm(curr_dir)
                
                alignment = np.dot(prev_dir, curr_dir)
                turn_penalty = 0.3 * (1.0 - alignment) / 2.0
        
        # 5. Stay bonus in low wind
        stay_bonus = 0.0
        if action == 8 and self.last_efficiency < 0.2:
            stay_bonus = 0.2
        
        # Combined shaped reward
        shaped_reward = (
            reward 
            + efficiency_bonus 
            + vmg_bonus 
            - step_penalty 
            - turn_penalty 
            + stay_bonus
        )
        
        # ========================================
        # EXPECTED SARSA UPDATE (key difference!)
        # ========================================
        # Instead of Q(s', a'), compute E[Q(s', ·)]
        
        # Get Q-values for next state
        next_q_values = self.q_table[next_state]
        
        # Compute policy probabilities for next state (epsilon-greedy)
        # We need to know which actions are valid in next_state
        # For simplicity, assume all 9 actions could be valid
        # In practice, you'd need the next observation to check physics
        
        # Epsilon-greedy policy probabilities
        num_actions = 9
        greedy_action = np.argmax(next_q_values)
        
        # Policy: ε/|A| for all actions, plus (1-ε) for greedy action
        policy_probs = np.ones(num_actions) * (self.exploration_rate / num_actions)
        policy_probs[greedy_action] += (1.0 - self.exploration_rate)
        
        # Expected value: E[Q(s', ·)] = Σ π(a'|s') Q(s', a')
        expected_next_value = np.dot(policy_probs, next_q_values)
        
        # TD target with expected value
        td_target = shaped_reward + self.discount_factor * expected_next_value
        td_error = td_target - self.q_table[state][action]
        
        self.q_table[state][action] += self.learning_rate * td_error

    def reset(self):
        """Reset episode-level state."""
        self.last_position = None
        self.last_action = None
        self.last_vmg = 0.0
        self.last_efficiency = 0.0
        
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
