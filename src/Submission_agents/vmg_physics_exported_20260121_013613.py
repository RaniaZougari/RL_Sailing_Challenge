"""
Q-Learning Agent for the Sailing Challenge - Trained Model

This file contains a Q-learning agent trained on the sailing environment.
The agent uses a discretized state space and a Q-table for decision making.
Auto-generated from: /home/unmars/Downloads/RL_Sailing_Challenge/src/agents/vmg_physics.py
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

        self.learning_rate = 0.15         # alpha - start high
        self.min_learning_rate = 0.05    # minimum alpha
        self.lr_decay_rate = 0.999       # decay per episode
        self.discount_factor = 0.99  # gamma
        self.exploration_rate = 0.09      # Start high for exploration
        self.min_exploration = 0.01      # Minimum epsilon
        self.eps_decay_rate = 0.995      # Decay per episode (slower)
        self.position_bins = 10
        self.velocity_bins = 5
        self.wind_bins = 8
        self.wind_preview_steps = 3
        self.grid_size = 32
        self.goal_position = np.array([16, 31])
        self.q_init_high = 10.0
        self.last_efficiency = 0.0
        self.last_vmg = 0.0

        # Q-table with learned values
        self.q_table = {}
        self._init_q_table()
    
    def _init_q_table(self):
        """Initialize the Q-table with learned values."""
        self.q_table[(0, 0)] = np.array([202.7407, 250.0671, 289.4394, 241.1139, 233.4031, 317.9432, 227.5042,
 232.6042, 239.0958])
        self.q_table[(0, 2)] = np.array([326.7859, 308.7088, 326.2932, 376.7354, 316.829 , 313.4456, 321.0151,
 317.4767, 316.7821])
        self.q_table[(0, 1)] = np.array([377.5825, 316.8268, 364.9576, 317.7968, 314.7091, 319.0001, 326.8104,
 308.1837, 310.2158])
        self.q_table[(15, 1)] = np.array([331.7566, 430.8249, 301.7226, 296.619 , 326.9957, 320.1205, 343.1373,
 391.0372, 320.1291])
        self.q_table[(15, 0)] = np.array([ 90.5126, 220.3776, 277.2767, 234.3806, 220.4154, 220.3808, 324.5379,
 225.6254, 218.837 ])
        self.q_table[(15, 2)] = np.array([314.309 , 321.6594, 285.6346, 314.1668, 305.5899, 295.032 , 296.8449,
 323.6703, 397.5533])
        self.q_table[(14, 2)] = np.array([407.0008, 496.9339, 286.2383, 285.4611, 341.4818, 333.1322, 342.6928,
 335.9591, 339.4723])
        self.q_table[(14, 1)] = np.array([436.8629, 287.5017, 235.8762, 288.2783, 298.8532, 281.9791, 340.3559,
 305.9201, 283.3613])
        self.q_table[(1, 1)] = np.array([323.2978, 332.1759, 388.8531, 375.398 , 324.7896, 324.8887, 328.3696,
 326.2766, 291.3748])
        self.q_table[(1, 2)] = np.array([353.4605, 334.1085, 351.6106, 326.2214, 327.166 , 332.266 , 334.7642,
 326.5912, 328.0242])
        self.q_table[(13, 1)] = np.array([363.843 , 595.9735, 182.044 , 317.296 , 301.3626, 468.1757, 333.8436,
 300.3086, 321.4173])
        self.q_table[(2, 1)] = np.array([360.5017, 356.4122, 364.9948, 355.2954, 352.4067, 356.8841, 368.9166,
 363.1872, 359.9184])
        self.q_table[(1, 0)] = np.array([114.669 , 184.0275, 145.2829, 341.3207, 239.6947, 132.1624, 227.8627,
 165.6348, 225.1376])
        self.q_table[(2, 2)] = np.array([369.7581, 367.8925, 357.4076, 360.9729, 350.4611, 352.283 , 349.285 ,
 354.1006, 356.0729])
        self.q_table[(13, 2)] = np.array([447.4083, 553.2714, 431.9161, 306.6257, 346.4088, 438.1188, 407.9364,
 385.2142, 447.1586])
        self.q_table[(12, 2)] = np.array([538.1219, 645.9296, 380.446 , 327.2791, 391.6439, 424.6872, 425.2547,
 462.3083, 402.9136])
        self.q_table[(12, 1)] = np.array([340.1222, 439.0702, 150.6858, 298.0305, 309.6556, 303.338 , 355.9409,
 538.4567, 335.373 ])
        self.q_table[(14, 0)] = np.array([ 44.6193,   0.9402,  35.747 ,  36.5951,  51.8228, 193.2682, 314.333 ,
  80.1215,   5.957 ])
        self.q_table[(3, 2)] = np.array([338.1298, 352.9384, 346.5482, 344.6836, 349.7765, 338.4433, 348.9385,
 388.8532, 353.9567])
        self.q_table[(3, 1)] = np.array([321.1149, 332.4875, 361.8419, 348.4697, 342.5307, 349.3055, 342.8699,
 345.5985, 306.1861])
        self.q_table[(12, 0)] = np.array([244.4715,   5.6739,   9.5747, 122.6326, 106.9467,  98.5245, 377.6845,
  93.9503,  49.9082])
        self.q_table[(4, 1)] = np.array([182.346 , 310.7897, 271.7777, 268.7016, 218.3797, 213.134 , 172.4246,
 269.5014, 165.8972])
        self.q_table[(4, 2)] = np.array([239.6054, 279.8312, 269.5878, 313.8348, 147.3991, 261.8896, 245.5061,
 367.0152, 272.1765])
        self.q_table[(5, 2)] = np.array([158.9068, 238.9213, 213.4798, 101.8587,  82.7799, 197.6189, 403.488 ,
 186.0708, 237.6602])
        self.q_table[(5, 1)] = np.array([102.0687, 190.478 , 181.0482, 287.0784,  77.2656, 170.2616, 169.4694,
 360.6117, 148.3658])
        self.q_table[(6, 2)] = np.array([283.0722, 365.9135, 238.7632, 170.2098, 132.4301, 255.7771, 226.1547,
 522.6635, 295.7926])
        self.q_table[(6, 1)] = np.array([336.5723, 474.5474, 291.693 , 177.5094, 217.3825, 166.5286, 232.6217,
 379.9344, 282.8373])
        self.q_table[(7, 2)] = np.array([492.6055, 489.1569, 497.0214, 431.5711, 185.6607, 376.2349, 488.13  ,
 577.1142, 424.9786])
        self.q_table[(8, 2)] = np.array([564.2173, 557.1657, 536.7296, 375.5181, 304.1386, 462.0174, 518.0042,
 525.5245, 628.3192])
        self.q_table[(8, 1)] = np.array([615.2817, 560.0075, 564.2078, 537.1239, 407.3904, 553.6161, 544.141 ,
 580.5461, 521.4715])
        self.q_table[(9, 1)] = np.array([643.6782, 616.2381, 597.8702, 418.7943, 144.3891, 643.1359, 649.2588,
 682.4049, 597.7219])
        self.q_table[(10, 1)] = np.array([668.1943, 698.2544, 638.1804, 391.0257, 532.3058, 609.3561, 675.0949,
 646.2338, 592.7239])
        self.q_table[(9, 2)] = np.array([666.8895, 585.9826, 593.836 , 324.5844, 359.3851, 599.942 , 629.0885,
 606.3216, 621.2017])
        self.q_table[(10, 2)] = np.array([681.6955, 665.3607, 551.3903, 332.3816, 667.6082, 656.7057, 671.9394,
 685.2661, 662.5854])
        self.q_table[(11, 2)] = np.array([636.1576, 665.9366, 476.1567, 224.852 , 540.8032, 621.1903, 628.7566,
 569.494 , 626.375 ])
        self.q_table[(11, 1)] = np.array([566.0908, 664.7039, 417.277 ,  70.4568, 509.1306, 647.2892, 522.8479,
 517.1488, 504.4694])
        self.q_table[(7, 1)] = np.array([291.4846, 421.3627, 348.7733, 428.4683, 122.2678, 319.1675, 386.6815,
 572.1481, 433.3024])
        self.q_table[(8, 0)] = np.array([ 35.5196, 212.6957, 624.1426,   1.5052,  33.1698, 141.5864, 376.2257,
 175.2308,  97.7753])
        self.q_table[(7, 0)] = np.array([219.515 ,  71.9992, 552.8649, 174.2327,   5.1162,   3.2985, 141.9541,
 140.5604, 160.085 ])
        self.q_table[(11, 0)] = np.array([ 80.6276, 501.2685,   2.7921,   0.9725, 270.3968, 110.1296, 108.6216,
  60.2835, 107.3059])
        self.q_table[(3, 0)] = np.array([ 37.9563, 258.0823,   5.7339, 123.2231,  42.2812,   3.2534,   5.4928,
 351.0184,  48.2092])
        self.q_table[(13, 0)] = np.array([ 83.3673,  61.6438,   8.9742, 143.3713, 178.3377, 404.8979, 209.0619,
 255.3219, 217.7897])
        self.q_table[(4, 0)] = np.array([  3.7969,   0.5625, 193.2765,   9.1556,   9.5625,   8.1085,   5.0578,
   4.3089,   3.8699])
        self.q_table[(6, 0)] = np.array([202.2508, 133.4575,  98.8452, 163.7778,  53.4926,   8.1116, 471.7553,
 177.4472, 165.6151])
        self.q_table[(2, 0)] = np.array([ 81.7699,  62.1573,  69.9252, 362.9237,  51.0463,   7.9496, 219.6894,
  57.7319, 127.1524])
        self.q_table[(5, 0)] = np.array([  1.9231,   4.952 , 146.7092,   5.8037,  25.7909,   1.7093,   0.7066,
   4.9812,   4.0517])
        self.q_table[(10, 0)] = np.array([113.0574, 721.654 ,   7.3383,   6.1171, 258.5874, 266.2854, 230.639 ,
 215.431 , 214.6755])
        self.q_table[(9, 0)] = np.array([209.3645,  21.367 ,   6.1452,  59.2296,   1.2076, 150.8282, 662.5403,
 229.1185, 228.0697])

    def discretize_state(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5] # Vent Local (Indice Prof)

        # 1. Angle du vent relatif au bateau (0 à 360)
        # On simplifie : on veut juste l'angle du vent
        wind_angle = np.arctan2(wy, wx)

        # 2. Angle vers le but relatif
        to_goal_x = self.goal_position[0] - x
        to_goal_y = self.goal_position[1] - y
        goal_angle = np.arctan2(to_goal_y, to_goal_x)

        # 3. Angle relatif (Différence entre vent et direction du but)
        # C'est LA métrique clé pour la voile (au près, au largue, etc.)
        relative_angle = (wind_angle - goal_angle + np.pi) % (2 * np.pi) - np.pi

        # Discretisation
        # Angle relatif en 16 secteurs (plus précis que 8)
        rel_angle_bin = int((relative_angle + np.pi) / (2 * np.pi) * 16) % 16

        # Vitesse (important pour savoir si on plane ou si on est à l'arrêt)
        v_mag = np.sqrt(vx**2 + vy**2)
        v_bin = 0 if v_mag < 0.2 else (1 if v_mag < 1.0 else 2)

        # On n'inclut PAS x et y => Table beaucoup plus petite => Apprentissage 10x plus rapide
        return (rel_angle_bin, v_bin)

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

    def _action_to_direction(self, action):
        """Convert action index to direction vector."""
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1),
            (0, 0)
        ]
        return np.array(directions[action])

    def _calculate_vmg(self, observation, action):
        """
        VMG intelligent pondéré par l'efficacité de navigation.

        Combine:
        - La direction vers le goal (cos_to_goal)
        - L'efficacité de navigation avec le vent local (polar diagram)

        Returns a value between 0 and 1.
        """
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
