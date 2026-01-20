"""
Q-Learning Agent for the Sailing Challenge - Trained Model

This file contains a Q-learning agent trained on the sailing environment.
The agent uses a discretized state space and a Q-table for decision making.
Auto-generated from: /home/unmars/Downloads/RL_Sailing_Challenge/src/agents/vmg_agent_v4.py
"""

import numpy as np
from agents.base_agent import BaseAgent
from src.sailing_physics import calculate_sailing_efficiency

class MyAgentTrained(BaseAgent):
    """
    A Q-learning agent trained on the sailing environment.
    Uses a discretized state space and a lookup table for actions.
    """
    
    def __init__(self):
        """Initialize the agent with the trained Q-table."""
        super().__init__()
        self.np_random = np.random.default_rng()

        self.INITIAL_EXPLORATION_RATE = 0.20
        self.INITIAL_LEARNING_RATE = 0.35
        self.learning_rate = self.INITIAL_LEARNING_RATE         # alpha - start high
        self.min_learning_rate = 0.05    # minimum alpha
        self.lr_decay_rate = 0.999       # decay per episode
        self.discount_factor = 0.99  # gamma
        self.exploration_rate = self.INITIAL_EXPLORATION_RATE      # Start high for exploration
        self.min_exploration = 0.01      # Minimum epsilon
        self.eps_decay_rate = 0.995      # Decay per episode (slower)
        self.position_bins = 10
        self.velocity_bins = 5
        self.wind_bins = 8
        self.wind_preview_steps = 3
        self.grid_size = 32
        self.goal_position = np.array([16, 31])
        self.q_init_high = 5.0  # Reduced from 10.0 to avoid noise
        self.last_position = None
        self.last_action = None
        self.last_efficiency = 0.0
        self.last_vmg = 0.0
        self.low_wind_threshold = 0.3  # Below this, special handling kicks in
        self.epsilon = 1e-6  # Numerical stability

        # Q-table with learned values
        self.q_table = {}
        self._init_q_table()
    
    def _init_q_table(self):
        """Initialize the Q-table with learned values."""
        self.q_table[(5, 2)] = np.array([558.9242, 592.6098, 542.789 , 511.1266, 536.6613, 536.877 , 519.5295,
 541.692 ,   0.6406])
        self.q_table[(6, 2)] = np.array([362.8189, 635.7844, 518.1874, 549.8061, 571.7304,   2.7729, 252.3164,
 530.4135,   3.1583])
        self.q_table[(6, 1)] = np.array([744.4603, 796.7297, 742.2442, 625.8642, 645.549 ,   1.3349, 562.7849,
 634.2775,   2.1946])
        self.q_table[(5, 1)] = np.array([622.3543, 610.7685, 637.5648, 538.9126, 536.2122, 603.9987, 736.3595,
 624.414 ,   3.8432])
        self.q_table[(7, 1)] = np.array([726.0281, 826.8144, 552.2289, 811.1842,   1.2798, 803.0296, 426.551 ,
 611.5632,   2.1755])
        self.q_table[(8, 1)] = np.array([7.2392e+02, 5.2510e+02, 5.9303e+02, 8.8759e+02, 1.5796e+00, 6.4698e+02,
 6.0351e+02, 8.2114e+02, 4.7233e-01])
        self.q_table[(9, 1)] = np.array([712.6477, 487.2377, 701.2257,   1.1883, 648.0959, 656.5082, 650.3859,
 776.5831,   2.9516])
        self.q_table[(4, 1)] = np.array([588.9002, 650.4545, 572.1542, 342.7718, 274.7861, 364.7328, 572.4302,
 748.5778,   1.5124])
        self.q_table[(7, 2)] = np.array([350.7509, 471.6705, 478.1506, 544.1857,   2.5823, 448.9377, 451.6335,
 514.8799,   1.5916])
        self.q_table[(4, 2)] = np.array([528.6271, 535.6525, 571.6874, 456.6959, 532.5903, 535.8001, 537.4404,
 548.395 ,   3.0121])
        self.q_table[(10, 1)] = np.array([752.4525, 656.9762, 585.445 , 529.5895, 565.696 , 535.4486, 597.8019,
 455.7367,   4.9783])
        self.q_table[(3, 1)] = np.array([140.0541, 327.8938, 236.6851, 423.6375,  97.2299,   3.0516, 691.1007,
 310.8822,   3.2342])
        self.q_table[(2, 1)] = np.array([223.2623, 285.5696, 117.5624,  -1.2037,  55.7295,   1.9192, 253.5171,
 613.2773,   1.8829])
        self.q_table[(8, 2)] = np.array([452.1279, 512.1832, 555.9628, 675.7293,   3.2213, 396.6904, 498.9253,
 499.6751,   1.1189])
        self.q_table[(9, 2)] = np.array([503.7453, 523.9129, 529.8715,   2.202 , 479.8335, 549.5921, 516.0258,
 491.3906,   2.393 ])
        self.q_table[(10, 2)] = np.array([605.9178, 553.3649, 525.2798, 523.7932, 527.7041, 524.6208, 521.0472,
 645.6438,   3.5048])
        self.q_table[(11, 2)] = np.array([593.6444, 474.1115, 495.827 , 497.1331, 492.6222, 490.6375, 364.7224,
 485.1382,   3.9918])
        self.q_table[(11, 1)] = np.array([429.1233, 362.6966, 658.778 , 230.1308, 285.3554, 370.7769, 296.1428,
 570.9824,   2.3868])
        self.q_table[(12, 1)] = np.array([117.5324, 288.0849, 188.1306, 591.1413, 190.8891,   1.943 ,   2.6415,
 484.6575,   3.0841])
        self.q_table[(3, 2)] = np.array([549.9259, 513.2181, 511.8076, 484.4011, 512.1518, 506.23  , 512.9453,
 567.1659,   1.7368])
        self.q_table[(12, 2)] = np.array([445.4843, 592.2915, 370.5563, 320.5946, 339.9966, 424.035 , 368.5988,
 393.7353,   1.1577])
        self.q_table[(13, 2)] = np.array([270.7643, 291.4247, 435.073 , 280.9552, 214.5019, 233.3112, 207.6064,
 456.9515,   1.8133])
        self.q_table[(14, 2)] = np.array([277.6922, 259.9878, 394.5723, 265.691 , 250.6188, 271.198 , 254.609 ,
 465.5289,   3.5829])
        self.q_table[(15, 2)] = np.array([325.909 , 398.5344, 318.6148, 327.1125, 284.3104, 177.9493, 509.2115,
 304.8619,   4.7068])
        self.q_table[(0, 2)] = np.array([426.7589, 321.1763, 382.1997, 399.7779, 226.8175, 528.597 , 425.4514,
 415.7588,   1.0895])
        self.q_table[(15, 1)] = np.array([ 17.0528,  56.4333, 189.981 ,  80.4068,   1.2257,   1.7563, 499.9472,
 405.608 ,   0.5695])
        self.q_table[(14, 1)] = np.array([281.8783,  98.555 , 134.4813,  91.2599, 396.2514,  89.7682,  52.2772,
   6.0956,   4.1166])
        self.q_table[(1, 1)] = np.array([242.6173,  18.5799, 485.0802, 101.5105, 239.3462, 240.3086, 568.7796,
   1.1586,   1.8252])
        self.q_table[(13, 1)] = np.array([228.9996, 462.3949, 108.835 ,  -2.1103,  77.5329,  81.8673,  41.4414,
  47.4909,   3.2666])
        self.q_table[(0, 1)] = np.array([1.2119e+02, 1.1121e+02, 2.1270e+02, 1.1920e+01, 2.3506e+02, 2.2303e+02,
 2.3803e+01, 4.6541e+02, 7.9612e-02])
        self.q_table[(3, 0)] = np.array([4.299 , 0.2288, 2.2514, 0.0688, 2.6187, 2.5918, 0.7857, 1.667 , 4.4525])
        self.q_table[(1, 2)] = np.array([427.3159, 453.2216, 419.2559, 340.1181, 430.9191, 535.2301, 426.0479,
 427.253 ,   4.7109])
        self.q_table[(2, 2)] = np.array([4.5946e+02, 4.5882e+02, 3.5644e+02, 4.9276e+02, 5.0298e+02, 5.4064e+02,
 4.6048e+02, 4.7882e+02, 3.3050e-01])
        self.q_table[(14, 0)] = np.array([4.3405, 4.959 , 4.5484, 3.3618, 1.6159, 3.0204, 1.0126, 2.8365, 4.475 ])
        self.q_table[(8, 0)] = np.array([2.5638, 3.2613, 4.2335, 2.1768, 0.0109, 0.9546, 0.4503, 0.9764, 3.3051])
        self.q_table[(9, 0)] = np.array([4.2481, 2.898 , 2.2192, 3.6121, 1.8114, 2.4179, 2.1159, 3.4011, 4.3499])
        self.q_table[(1, 0)] = np.array([0.2936, 1.6546, 1.1313, 1.62  , 0.4242, 3.967 , 4.5342, 4.467 , 1.5891])
        self.q_table[(2, 0)] = np.array([0.286 , 0.018 , 1.4201, 0.3521, 0.2151, 4.5402, 2.4761, 0.41  , 1.2697])
        self.q_table[(4, 0)] = np.array([4.2002, 0.2528, 2.5698, 0.0752, 1.7965, 3.5822, 4.1492, 1.3577, 4.9863])

    def discretize_state(self, observation):
        # 1. Récupérer les infos relatives
        boat_pos = np.array([observation[0], observation[1]])
        goal_pos = self.goal_position # Assure-toi que self.goal_position est défini (c'est [16, 31] souvent)
        wind_vec = np.array([observation[4], observation[5]])

        # Vecteur vers le but
        to_goal = goal_pos - boat_pos
        dist_to_goal = np.linalg.norm(to_goal)

        # Angle du but (0 à 2pi)
        angle_goal = np.arctan2(to_goal[1], to_goal[0])

        # Angle du vent (0 à 2pi)
        angle_wind = np.arctan2(wind_vec[1], wind_vec[0])

        # 2. Angle RELATIF : Le vent vient-il de face par rapport à mon but ?
        # C'est LA donnée cruciale.
        # Si c'est proche de PI (180°), le vent vient de l'objectif -> Il faut tirer des bords.
        rel_angle = (angle_wind - angle_goal) % (2 * np.pi)

        # Discrétisation plus fine pour l'angle relatif (16 secteurs)
        # Cela permet de bien distinguer "Face au vent" de "Légèrement de travers"
        angle_bin = int(rel_angle / (2 * np.pi) * 16) % 16

        # Distance (logarithmique pour avoir plus de précision près du but)
        if dist_to_goal < 1:
            dist_bin = 0
        elif dist_to_goal < 5:
            dist_bin = 1
        else:
            dist_bin = 2

        # On retourne un tuple simple. Plus l'état est petit, plus il apprend vite.
        return (angle_bin, dist_bin)

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
        # cos(135 deg) = -0.707. We want to avoid angles > 135 deg relative to wind vector (which points downwind)
        no_go_threshold = -0.707

        # Check if Goal is in No-Go Zone
        force_tacking = False
        if wind_mag > self.epsilon:
            # Vector to goal
            goal_vec = self.goal_position - current_position
            goal_dist = np.linalg.norm(goal_vec)

            if goal_dist > self.epsilon:
                goal_dir = goal_vec / goal_dist
                # Cosine of angle between wind (blowing TO) and goal direction
                cos_goal_wind = (goal_dir[0] * wx + goal_dir[1] * wy) / wind_mag

                # If goal is "upwind" (angle > 135 degrees relative to wind flow)
                # This corresponds to cos < -0.707
                if cos_goal_wind < no_go_threshold:
                    force_tacking = True

        if force_tacking:
            # Goal is in No-Go Zone -> Force Tacking
            # We want to sail at approx 45 degrees to the wind (coming FROM)
            # which is 135 degrees relative to wind vector (going TO).
            # We identify the two best actions (Port and Starboard tacks)

            best_actions = []

            # Wind direction angle
            wind_angle = np.arctan2(wy, wx)

            # Ideal tacking angles: Wind angle +/- 135 degrees
            ideal_tack_1 = wind_angle + 3 * np.pi / 4
            ideal_tack_2 = wind_angle - 3 * np.pi / 4

            # Normalize angles
            ideal_tack_1 = (ideal_tack_1 + np.pi) % (2 * np.pi) - np.pi
            ideal_tack_2 = (ideal_tack_2 + np.pi) % (2 * np.pi) - np.pi

            # Find closest discrete actions to these ideal angles
            min_diff_1 = float('inf')
            best_action_1 = -1
            min_diff_2 = float('inf')
            best_action_2 = -1

            for i in range(8):
                ax, ay = action_vectors[i]
                action_angle = np.arctan2(ay, ax)

                # Difference for tack 1
                diff1 = abs(np.arctan2(np.sin(action_angle - ideal_tack_1), np.cos(action_angle - ideal_tack_1)))
                if diff1 < min_diff_1:
                    min_diff_1 = diff1
                    best_action_1 = i

                # Difference for tack 2
                diff2 = abs(np.arctan2(np.sin(action_angle - ideal_tack_2), np.cos(action_angle - ideal_tack_2)))
                if diff2 < min_diff_2:
                    min_diff_2 = diff2
                    best_action_2 = i

            if best_action_1 != -1:
                valid_actions.append(best_action_1)
            if best_action_2 != -1 and best_action_2 != best_action_1:
                valid_actions.append(best_action_2)

        else:
            # Standard filtering (prevent sailing directly into wind, but allow other moves)
            for i in range(8):
                ax, ay = action_vectors[i]
                a_mag = np.sqrt(ax**2 + ay**2)

                if wind_mag > self.epsilon:
                    cos_theta = (ax * wx + ay * wy) / (a_mag * wind_mag)

                    if cos_theta >= no_go_threshold:
                        valid_actions.append(i)
                else:
                    # No wind: all directions valid
                    valid_actions.append(i)


        # --- Epsilon-greedy ---
        if self.np_random.random() < self.exploration_rate:
            if not valid_actions: # Fallback if empty (shouldn't happen)
                 valid_actions = list(range(9))
            action = self.np_random.choice(valid_actions)
        else:
            q_values = self.q_table[state].copy()
            mask = np.full(9, -np.inf)
            # Ensure valid_actions are within bounds
            safe_valid_actions = [a for a in valid_actions if 0 <= a < 9]
            if not safe_valid_actions:
                 safe_valid_actions = [8] # Fallback to stay
            mask[safe_valid_actions] = 0
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

        # --- Reward shaping components ---

        # 1. Efficiency bonus (polar diagram) - keep moderate
        efficiency_bonus = self.last_efficiency * 0.5

        # 2. VMG bonus - REDUCED from 10.0 to 2.0 (key fix!)
        vmg_bonus = self.last_vmg * 2.0

        # 3. Step penalty - INCREASED to discourage long trajectories
        step_penalty = 2

        # Combined shaped reward
        shaped_reward = (
            reward
            + efficiency_bonus
            + vmg_bonus
            - step_penalty
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

        self.exploration_rate = self.INITIAL_EXPLORATION_RATE
        self.learning_rate = self.INITIAL_LEARNING_RATE

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

    def _action_to_direction(self, action):
        """Convert action index to direction vector."""
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1),
            (0, 0)
        ]
        return np.array(directions[action])

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
