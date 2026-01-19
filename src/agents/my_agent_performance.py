
import numpy as np
from collections import defaultdict
from agents.base_agent import BaseAgent


class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()

        # Paramètres d'apprentissage - OPTIMISÉS
        self.learning_rate = 0.2  # Plus élevé pour apprendre plus vite
        self.discount_factor = 0.99
        self.exploration_rate = 0.5  # Plus d'exploration initiale
        self.exploration_decay = 0.995
        self.min_exploration = 0.01

        # Discrétisation - AFFINÉE
        self.position_bins = 12  # Plus de précision
        self.velocity_bins = 6   # Plus de précision
        self.wind_bins = 8
        self.wind_preview_steps = 4  # Voir plus loin
        self.grid_size = 32

        # Physique de voile - CALIBRÉE selon vos graphiques
        self.NO_GO_ANGLE = np.radians(40)      # Zone interdite (légèrement réduite)
        self.OPTIMAL_ANGLE = np.radians(90)    # Beam reach optimal
        
        # Q-table avec initialisation optimiste
        self.q_table = {}
        self.initial_q_value = 50.0  # Très optimiste pour encourager exploration
        
        # SARSA
        self.last_state = None
        self.last_action = None
        
        # Compteurs de visite pour adaptive learning - FIXÉ avec defaultdict
        self.visit_counts = defaultdict(int)  # ✅ Retourne 0 par défaut
        
        # Poids physique ADAPTATIF (commence bas, augmente si Q-learning échoue)
        self.physics_weight = 0.2  # Commence à 20% seulement
        self.min_physics_weight = 0.2
        self.max_physics_weight = 0.6
        
        # Tracking pour ajuster physics_weight
        self.recent_rewards = []
        self.episode_count = 0

    def get_sailing_efficiency(self, wind_angle):
        """
        Efficacité de navigation PRÉCISE selon vos graphiques.
        """
        abs_angle = abs(wind_angle)
        
        # No-Go Zone (0-40°) - TRÈS pénalisé
        if abs_angle < self.NO_GO_ANGLE:
            return 0.02
        
        # Close-Hauled (40-60°) - Acceptable mais pas optimal
        elif abs_angle < np.radians(60):
            t = (abs_angle - self.NO_GO_ANGLE) / (np.radians(60) - self.NO_GO_ANGLE)
            return 0.5 + t * 0.2  # 0.5 -> 0.7
        
        # Beam Reach (60-120°) - ZONE OPTIMALE
        elif abs_angle < np.radians(120):
            if abs_angle < self.OPTIMAL_ANGLE:
                # Montée vers l'optimal
                t = (abs_angle - np.radians(60)) / (self.OPTIMAL_ANGLE - np.radians(60))
                return 0.7 + t * 0.3  # 0.7 -> 1.0
            else:
                # Descente après l'optimal
                t = (abs_angle - self.OPTIMAL_ANGLE) / (np.radians(120) - self.OPTIMAL_ANGLE)
                return 1.0 - t * 0.25  # 1.0 -> 0.75
        
        # Broad Reach (120-150°)
        elif abs_angle < np.radians(150):
            t = (abs_angle - np.radians(120)) / (np.radians(150) - np.radians(120))
            return 0.75 - t * 0.25  # 0.75 -> 0.5
        
        # Running (150-180°)
        else:
            return 0.5

    def should_tack(self, observation):
        """
        Détermine si on doit tacker (changer de bord) pour éviter le no-go zone.
        CRUCIAL pour minimiser les steps !
        """
        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5]
        
        goal_x, goal_y = 16, 31
        dx, dy = goal_x - x, goal_y - y
        
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return False, 0  # Au but
        
        goal_angle = np.arctan2(dy, dx)
        wind_angle = np.arctan2(wy, wx)
        
        # Angle relatif au vent si on va vers le but
        relative_angle = ((goal_angle - wind_angle + np.pi) % (2*np.pi)) - np.pi
        
        # Si dans no-go zone, calculer meilleur angle de tack
        if abs(relative_angle) < self.NO_GO_ANGLE:
            # Tacker à 50° du vent (juste au-dessus du no-go)
            tack_angle = np.radians(50)
            if relative_angle > 0:
                # Tack à tribord
                return True, wind_angle + tack_angle
            else:
                # Tack à bâbord
                return True, wind_angle - tack_angle
        
        return False, goal_angle

    def get_action_from_angle(self, target_angle):
        """
        Convertit un angle cible en action (0-7).
        """
        # Normaliser l'angle
        target_angle = ((target_angle + np.pi) % (2*np.pi)) - np.pi
        
        # Actions correspondent à 8 directions
        action_angles = [
            np.radians(90),   # 0: North
            np.radians(45),   # 1: NE
            np.radians(0),    # 2: East
            np.radians(-45),  # 3: SE
            np.radians(-90),  # 4: South
            np.radians(-135), # 5: SW
            np.radians(180),  # 6: West
            np.radians(135),  # 7: NW
        ]
        
        # Trouver l'action la plus proche
        best_action = 0
        min_diff = float('inf')
        
        for i, angle in enumerate(action_angles):
            diff = abs(((target_angle - angle + np.pi) % (2*np.pi)) - np.pi)
            if diff < min_diff:
                min_diff = diff
                best_action = i
        
        return best_action

    def get_physics_action(self, observation):
        """
        Action optimale selon la physique PURE.
        Utilisé pour guider l'exploration et comme fallback.
        """
        x, y = observation[0], observation[1]
        goal_x, goal_y = 16, 31
        
        # Si au but, rester
        if abs(goal_x - x) < 0.5 and abs(goal_y - y) < 0.5:
            return 8
        
        # Check si besoin de tacker
        need_tack, target_angle = self.should_tack(observation)
        
        if need_tack:
            # Utiliser l'angle de tack
            return self.get_action_from_angle(target_angle)
        else:
            # Aller vers le but
            dx, dy = goal_x - x, goal_y - y
            goal_angle = np.arctan2(dy, dx)
            return self.get_action_from_angle(goal_angle)

    def get_physics_score(self, action, observation):
        """
        Score physique pour une action donnée.
        """
        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5]
        
        goal_x, goal_y = 16, 31
        dx, dy = goal_x - x, goal_y - y
        
        # Au but
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return 100.0 if action == 8 else 0.0
        
        goal_angle = np.arctan2(dy, dx)
        wind_angle = np.arctan2(wy, wx)
        
        # Angles des actions
        action_angles = [
            np.radians(90), np.radians(45), np.radians(0), np.radians(-45),
            np.radians(-90), np.radians(-135), np.radians(180), np.radians(135),
            goal_angle  # Action 8: stay
        ]
        
        action_angle = action_angles[action]
        
        # Score de direction (0 à 1)
        angle_to_goal = abs(((action_angle - goal_angle + np.pi) % (2*np.pi)) - np.pi)
        direction_score = np.cos(angle_to_goal)
        
        # Score d'efficacité (0 à 1)
        wind_relative = ((action_angle - wind_angle + np.pi) % (2*np.pi)) - np.pi
        efficiency_score = self.get_sailing_efficiency(wind_relative)
        
        # Pénalité pour "stay"
        if action == 8:
            return direction_score * 10  # Très faible
        

        return efficiency_score * 60 + direction_score * 40

    def discretize_state(self, observation):
        """Discrétisation affinée."""
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

        # Preview du vent (8 directions)
        wind_grid = wind_flattened.reshape(32, 32, 2)
        gx, gy = int(x), int(y)

        preview_offsets = [
            (-self.wind_preview_steps, 0), (self.wind_preview_steps, 0),
            (0, self.wind_preview_steps), (0, -self.wind_preview_steps),
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
        Sélection d'action OPTIMISÉE avec physique adaptative.
        """
        state = self.discretize_state(observation)

        # Initialiser avec valeur optimiste
        if state not in self.q_table:
            self.q_table[state] = np.full(9, self.initial_q_value)
        
        self.visit_counts[state] += 1

        # Epsilon-greedy avec GUIDANCE physique
        if self.np_random.random() < self.exploration_rate:
            # Exploration  : 70% physique, 30% aléatoire
            if self.np_random.random() < 0.7:
                action = self.get_physics_action(observation)
            else:
                action = self.np_random.integers(0, 9)
        else:
            # Exploitation : Q-values BOOSTÉES par physique
            q_values = self.q_table[state].copy()
            
            # Adaptive physics weight basé sur nombre de visites
            adaptive_weight = self.physics_weight
            if self.visit_counts[state] < 5:
                # Peu visité : plus de physique
                adaptive_weight = min(0.5, self.physics_weight + 0.2)
            
            # Boost basé sur physique
            for a in range(9):
                physics_score = self.get_physics_score(a, observation)
                q_values[a] = (1 - adaptive_weight) * q_values[a] + \
                              adaptive_weight * physics_score
            
            action = np.argmax(q_values)

        self.last_state = state
        self.last_action = action

        return action

    def learn(self, state, action, reward, next_state):
        """
        SARSA avec learning rate adaptatif et reward shaping.
        """
        if state not in self.q_table:
            self.q_table[state] = np.full(9, self.initial_q_value)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.full(9, self.initial_q_value)

        # Learning rate adaptatif (diminue avec visites) - FIXÉ
        visit_count = self.visit_counts[state]  # ✅ defaultdict retourne 0 si absent
        adaptive_lr = self.learning_rate / (1 + visit_count * 0.005)

        # Choisir next_action (SARSA)
        if self.np_random.random() < self.exploration_rate:
            next_action = self.np_random.integers(0, 9)
        else:
            next_action = np.argmax(self.q_table[next_state])

        # SARSA update
        td_target = reward + self.discount_factor * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += adaptive_lr * td_error

        # Decay exploration
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )
        
        # Track reward pour ajuster physics_weight
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)

    def reset(self):
        """Reset avec ajustement adaptatif du physics_weight."""
        self.last_state = None
        self.last_action = None
        self.episode_count += 1
        
        # Tous les 50 épisodes, évaluer et ajuster
        if self.episode_count % 50 == 0 and len(self.recent_rewards) > 50:
            avg_reward = np.mean(self.recent_rewards[-50:])
            
            # Si performance faible, augmenter physics_weight
            if avg_reward < 20:
                self.physics_weight = min(self.max_physics_weight, 
                                         self.physics_weight + 0.05)
            # Si bonne performance, diminuer physics_weight (laisser Q-learning dominer)
            elif avg_reward > 60:
                self.physics_weight = max(self.min_physics_weight,
                                         self.physics_weight - 0.05)
        
        # Boost exploration
        if self.episode_count % 100 == 0:
            self.exploration_rate = min(0.3, self.exploration_rate * 2)

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                'q_table': self.q_table,
                'exploration_rate': self.exploration_rate,
                'physics_weight': self.physics_weight,
                'visit_counts': dict(self.visit_counts), 
                'episode_count': self.episode_count
            }, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.exploration_rate = data.get('exploration_rate', 0.01)
            self.physics_weight = data.get('physics_weight', 0.2)
            self.visit_counts = defaultdict(int, data.get('visit_counts', {}))
            self.episode_count = data.get('episode_count', 0)