"""
High-performance Physics-Aware Q-Learning Agent
Optimized for VMG (Velocity Made Good) and Fast Convergence.
"""

import numpy as np
import pickle
from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        
        # --- Hyperparamètres d'apprentissage ---
        # Alpha plus bas pour éviter d'oublier les bonnes manœuvres
        self.alpha = 0.1    
        # Gamma haut car la voile est une stratégie long-terme
        self.gamma = 0.99   
        self.epsilon = 0.1  # Sera géré par le script de training
        
        # --- Paramètres Physiques ---
        self.no_go_angle = np.pi / 4  # 45 degrés
        
        # --- Q-Table ---
        # Utilisation d'un dictionnaire pour la flexibilité
        self.q_table = {}
        
        # RNG
        self.rng = np.random.default_rng()

    # ============================================================
    # 1. OUTILS PHYSIQUES & MATHS
    # ============================================================
    
    def _get_angle(self, v):
        """Retourne l'angle d'un vecteur en radians."""
        return np.arctan2(v[1], v[0])

    def _angle_diff(self, a, b):
        """Différence signée entre deux angles (-pi à pi)."""
        diff = a - b
        return (diff + np.pi) % (2 * np.pi) - np.pi

    # ============================================================
    # 2. DISCRÉTISATION INTELLIGENTE (STATE SPACE)
    # ============================================================
    
    def discretize_state(self, obs, info=None):
        """
        Transforme l'observation continue en état discret optimisé pour la voile.
        """
        # Récupération des données
        x, y = obs[0], obs[1]
        vx, vy = obs[2], obs[3]
        wx, wy = obs[4], obs[5]
        
        # Gestion robuste du Goal (Training vs Submission)
        if info is not None and "goal" in info:
            gx, gy = info["goal"]
        else:
            gx, gy = 16, 31 # Valeur par défaut Codabench
            
        # Vecteurs
        boat_vel = np.array([vx, vy])
        wind_vec = np.array([wx, wy])
        to_goal = np.array([gx - x, gy - y])
        
        # Angles absolus
        # Si la vitesse est trop faible, on utilise l'orientation du vecteur but temporairement
        # pour éviter le bruit, ou on garde le dernier cap connu (non dispo ici, donc approx).
        if np.linalg.norm(boat_vel) > 0.05:
            boat_angle = self._get_angle(boat_vel)
        else:
            # Si on est à l'arrêt, on considère qu'on regarde vers le but (approx)
            # ou on pourrait ajouter l'angle du gouvernail si dispo.
            boat_angle = 0.0 

        wind_angle = self._get_angle(wind_vec)
        goal_angle = self._get_angle(to_goal)
        
        # --- CRUCIAL : ANGLES RELATIFS ---
        
        # 1. Angle du vent par rapport au bateau (D'où vient le vent ?)
        # 0 = Vent de face, +/- pi = Vent arrière
        theta_wind_rel = self._angle_diff(wind_angle, boat_angle)
        
        # 2. Angle du but par rapport au bateau
        theta_goal_rel = self._angle_diff(goal_angle, boat_angle)
        
        # 3. Vitesse
        speed = np.linalg.norm(boat_vel)

        # --- BINNING (Discrétisation) ---
        
        # Binning du Vent (Situation d'allure) : 8 secteurs
        # 0: Face au vent (No-Go)
        # 1-3: Tribord (Droite)
        # 4-6: Bâbord (Gauche)
        # 7: Vent arrière
        wind_sector = int((theta_wind_rel + np.pi) / (2 * np.pi) * 8) % 8
        
        # Binning du But (Direction à prendre) : 8 secteurs
        goal_sector = int((theta_goal_rel + np.pi) / (2 * np.pi) * 8) % 8
        
        # Binning Vitesse : Indispensable pour savoir si on est "en panne"
        # 0: À l'arrêt (DANGER)
        # 1: Lent
        # 2: Rapide (Planning)
        if speed < 0.2:
            speed_bin = 0
        elif speed < 1.0:
            speed_bin = 1
        else:
            speed_bin = 2
            
        # L'état est un tuple compact
        return (wind_sector, goal_sector, speed_bin)

    # ============================================================
    # 3. PRISE DE DÉCISION
    # ============================================================

    def act(self, obs, info=None):
        state = self.discretize_state(obs, info)

        # Initialisation lazy de la Q-table
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9) # 9 actions possibles (rudders)

        # Epsilon-Greedy
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, 9)
        
        # Exploitation : on prend la meilleure action
        # Petite astuce : si plusieurs actions ont la même valeur max, on choisit au hasard parmi elles
        # pour éviter de rester bloqué sur l'action d'indice 0 au début.
        q_vals = self.q_table[state]
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(self.rng.choice(best_actions))

    # ============================================================
    # 4. APPRENTISSAGE & REWARD SHAPING (LE SECRET)
    # ============================================================

    def learn(self, obs, action, reward, next_obs, done, info=None):
        state = self.discretize_state(obs, info)
        next_state = self.discretize_state(next_obs, info)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)

        # --- CALCUL DE LA REWARD PERSONNALISÉE ---
        # On ignore la reward de base de l'env (souvent trop simple) 
        # et on recalcule une reward basée sur la physique.
        custom_reward = self._calculate_shaped_reward(obs, next_obs, done, info)

        # Mise à jour Q-Learning classique
        best_next_q = np.max(self.q_table[next_state])
        td_target = custom_reward + self.gamma * best_next_q * (1 - float(done))
        td_error = td_target - self.q_table[state][action]
        
        self.q_table[state][action] += self.alpha * td_error

    def _calculate_shaped_reward(self, obs, next_obs, done, info):
        """
        Calcule une reward "Dense" basée sur la VMG.
        C'est ici que se joue la performance.
        """
        # Récupération positions
        x, y = obs[0], obs[1]
        nx, ny = next_obs[0], next_obs[1]
        
        if info is not None and "goal" in info:
            gx, gy = info["goal"]
        else:
            gx, gy = 16, 31

        # Distances
        dist_prev = np.hypot(gx - x, gy - y)
        dist_curr = np.hypot(gx - nx, gy - ny)
        
        # 1. REWARD PRINCIPALE : VMG (Velocity Made Good)
        # On récompense massivement le fait de se rapprocher du but.
        # Facteur 100 pour que ce soit significatif par rapport au step penalty
        vmg_reward = (dist_prev - dist_curr) * 100.0
        
        # 2. PENALITÉ NO-GO ZONE (Vent de face)
        # Si le bateau pointe face au vent, c'est catastrophique.
        wx, wy = obs[4], obs[5]
        vx, vy = obs[2], obs[3]
        
        if np.linalg.norm([vx, vy]) > 0.01:
            boat_angle = np.arctan2(vy, vx)
            wind_angle = np.arctan2(wy, wx)
            angle_diff = abs(self._angle_diff(boat_angle, wind_angle))
            
            # Si on est dans le cone de 45° face au vent
            if angle_diff > (np.pi - self.no_go_angle): 
                # Note: Dans cet env, le vent est defini "vers où il souffle" ou "d'où il vient" ?
                # Standard meteo: d'où il vient. Standard vectoriel: vers où il va.
                # En général sailing_env: wind vector direction. 
                # Si wind vector (1,0), vent souffle vers l'Est.
                # Face au vent = aller vers (-1, 0). Angle diff ~ pi.
                vmg_reward -= 5.0 # Punition
        
        # 3. BONUS TERMINAL
        if done:
            return 200.0 # Gros bonus pour finir
            
        # 4. STEP PENALTY
        # Petite penalité pour inciter à aller vite
        vmg_reward -= 0.1
        
        return vmg_reward

    # ============================================================
    # 5. SAUVEGARDE / CHARGEMENT
    # ============================================================

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)

    def reset(self):
        pass
        
    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)