"""
Agent combining Q-Learning with physical sailing rules for improved navigation.
"""

import numpy as np
from agents.base_agent import BaseAgent


class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

        # Learning parameters
        self.learning_rate = 0.15
        self.discount_factor = 0.99
        self.exploration_rate = 0.4
        self.exploration_decay = 0.997
        self.min_exploration = 0.01

        # Discretization parameters
        self.position_bins = 10
        self.velocity_bins = 5
        self.wind_bins = 8
        self.wind_preview_steps = 3
        self.grid_size = 32

        # Physical sailing parameters
        self.NO_GO_ANGLE = np.radians(45)      # No-Go Zone
        self.BEAM_REACH_ANGLE = np.radians(90) # Optimal sailing angle
        self.optimal_efficiency = 1.0
        
        # Q-table
        self.q_table = {}
        self.initial_q_value = 10.0
        
        # SARSA
        self.last_state = None
        self.last_action = None
        
        # Weighting between Q-learning and physics
        self.physics_weight = 0.5  # 50% physics, 50% Q-learning

    def get_sailing_efficiency(self, wind_angle):
        """
        Calculate sailing efficiency based on wind angle.
        wind_angle: angle between boat direction and wind direction (radians)
        Returns a score between 0 and 1.
        """
        abs_angle = abs(wind_angle)
        
        # No-Go Zone (0-45°)
        if abs_angle < self.NO_GO_ANGLE:
            return 0.05  # Très inefficace
        
        # Close-Hauled (45-60°)
        elif abs_angle < np.radians(60):
            # Interpolation linéaire de 0.5 à 0.7
            t = (abs_angle - self.NO_GO_ANGLE) / (np.radians(60) - self.NO_GO_ANGLE)
            return 0.5 + t * 0.2
        
        # Beam Reach (60-120°) - optimal
        elif abs_angle < np.radians(120):
            # Interpolation de 0.7 à 1.0 à 90°, puis 1.0 à 0.7
            if abs_angle < self.BEAM_REACH_ANGLE:
                t = (abs_angle - np.radians(60)) / (self.BEAM_REACH_ANGLE - np.radians(60))
                return 0.7 + t * 0.3
            else:
                t = (abs_angle - self.BEAM_REACH_ANGLE) / (np.radians(120) - self.BEAM_REACH_ANGLE)
                return 1.0 - t * 0.3
        
        # Broad Reach (120-150°)
        elif abs_angle < np.radians(150):
            t = (abs_angle - np.radians(120)) / (np.radians(150) - np.radians(120))
            return 0.7 - t * 0.2
        
        # Running (150-180°)
        else:
            return 0.5

    def get_physics_score(self, action, observation):
        """
        Chooses an action based on physical sailing principles.
        """
        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5]
        
        # Direction towards the goal
        goal_x, goal_y = 16, 31
        dx = goal_x - x
        dy = goal_y - y
        
        if abs(dx) < 0.1 and abs(dy) < 0.1:
            return 1.0 if action == 8 else 0.0  # Stay if at goal
        
        goal_angle = np.arctan2(dy, dx)
        wind_angle = np.arctan2(wy, wx)
        
        # Action angles (radians)
        action_angles = [
            np.radians(90),   # 0: North
            np.radians(45),   # 1: NE
            np.radians(0),    # 2: East
            np.radians(-45),  # 3: SE
            np.radians(-90),  # 4: South
            np.radians(-135), # 5: SW
            np.radians(180),  # 6: West
            np.radians(135),  # 7: NW
            goal_angle        # 8: Stay (direction au but)
        ]
        
        action_angle = action_angles[action]
        
        # Scoring components
        angle_to_goal = abs(((action_angle - goal_angle + np.pi) % (2*np.pi)) - np.pi)
        direction_score = np.cos(angle_to_goal)  # 1 si aligné et 0 si perpendiculaire
        
        # Efficiency score based on wind angle
        wind_relative_angle = ((action_angle - wind_angle + np.pi) % (2*np.pi)) - np.pi
        efficiency_score = self.get_sailing_efficiency(wind_relative_angle)
        
        # Penality if staying
        stay_penalty = 0.5 if action == 8 else 1.0
        
        # Combine scores
        physics_score = (direction_score * 0.5 + efficiency_score * 0.5) * stay_penalty
        
        return physics_score

    def get_best_action_physics(self, observation):
        """
        Gets the best action purely based on physics scoring.
        """
        scores = np.array([self.get_physics_score(a, observation) for a in range(9)])
        return np.argmax(scores)

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

        wind_grid = wind_flattened.reshape(32, 32, 2)
        gx, gy = int(x), int(y)

        preview_offsets = [
            (-self.wind_preview_steps, 0),
            (self.wind_preview_steps, 0),
            (0, self.wind_preview_steps),
            (0, -self.wind_preview_steps),
            (-self.wind_preview_steps, self.wind_preview_steps),
            (-self.wind_preview_steps, -self.wind_preview_steps),
            (self.wind_preview_steps, self.wind_preview_steps),
            (self.wind_preview_steps, -self.wind_preview_steps),
        ]

        preview_bins = []
        for dx, dy in preview_offsets:
            xp = np.clip(gx + dx, 0, 31)
            yp = np.clip(gy + dy, 0, 31)
            wxp, wyp = wind_grid[xp, yp]
            ang = np.arctan2(wyp, wxp)
            preview_bins.append(
                int(((ang + np.pi) / (2 * np.pi)) * self.wind_bins) % self.wind_bins
            )

        return (x_bin, y_bin, v_bin, wind_bin, *preview_bins)

    def act(self, observation, info=None):
        """
        Select an action using epsilon-greedy strategy combined with physical sailing rules.
        """
        state = self.discretize_state(observation)

        if state not in self.q_table:
            self.q_table[state] = np.full(9, self.initial_q_value)

        # Epsilon-greedy with physics guidance
        if self.np_random.random() < self.exploration_rate:
            # Exploration : 50% Q-learning, 50% physique
            if self.np_random.random() < 0.5:
                action = self.get_best_action_physics(observation)
            else:
                action = self.np_random.integers(0, 9)
        else:
            # Exploitation : combine Q-values et physique
            q_values = self.q_table[state].copy()
            
            # Boost des Q-values selon la physique
            for a in range(9):
                physics_score = self.get_physics_score(a, observation)
                # Combine scores
                q_values[a] = (1 - self.physics_weight) * q_values[a] + \
                              self.physics_weight * physics_score * 100
            
            action = np.argmax(q_values)

        # SARSA
        self.last_state = state
        self.last_action = action

        return action

    def learn(self, state, action, reward, next_state):
        """
        SARSA avec bonus de récompense basé sur la physique.
        """
        if state not in self.q_table:
            self.q_table[state] = np.full(9, self.initial_q_value)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.full(9, self.initial_q_value)

        # Choose next action for SARSA
        if self.np_random.random() < self.exploration_rate:
            next_action = self.np_random.integers(0, 9)
        else:
            next_action = np.argmax(self.q_table[next_state])

        # SARSA update standard
        td_target = reward + self.discount_factor * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

        # less exploration over time
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

    def reset(self):
        self.last_state = None
        self.last_action = None

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