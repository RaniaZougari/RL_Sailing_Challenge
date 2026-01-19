"""
High-performance Physics-Aware Q-Learning Agent
VERSION V2: Optimized for VMG and Stability (Anti-Zigzag)
"""

import numpy as np
import pickle
from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        
        # --- Hyperparamètres ---
        # Alpha modéré pour ne pas osciller sur les valeurs Q
        self.alpha = 0.15    
        # Gamma très haut : on veut le chemin le plus court global
        self.gamma = 0.88  
        self.epsilon = 0.2  # Géré par le training script

        self.exploration_rate = 0.5
        
        # --- Paramètres Physiques ---
        self.no_go_angle = 0.70  # ~40-45 degrés (radians)
        
        # --- Discretization (Crucial) ---
        # On augmente la précision sur les angles pour trouver le "sweet spot" (close hauled)
        self.wind_bins = 12    # Finesse sur l'angle du vent
        self.goal_bins = 16    # Finesse sur le cap à suivre
        self.speed_bins = 3    # 0=Slow, 1=Medium, 2=Fast
        self.omega_bins = 3    # -1 (Turn Left), 0 (Stable), 1 (Turn Right)
        
        self.q_table = {}
        self.rng = np.random.default_rng()
        
        # Action précédente (pour penaliser le changement brutal)
        self.last_action = 4

    # ============================================================
    # 1. OUTILS MATHS
    # ============================================================
    
    def _angle_diff(self, a, b):
        """Différence signée entre deux angles (-pi à pi)."""
        diff = a - b
        return (diff + np.pi) % (2 * np.pi) - np.pi

    # ============================================================
    # 2. STATE SPACE RAFFINÉ
    # ============================================================
    
    def discretize_state(self, obs, info=None):
        """
        State = (AngleVentRelatif, AngleButRelatif, Vitesse, VitesseRotation)
        """
        x, y = obs[0], obs[1]
        vx, vy = obs[2], obs[3]
        wx, wy = obs[4], obs[5]
        
        # Goal handling
        if info is not None and "goal" in info:
            gx, gy = info["goal"]
        else:
            gx, gy = 16, 31 

        # --- Calculs vectoriels ---
        boat_vel = np.array([vx, vy])
        speed = np.linalg.norm(boat_vel)
        
        if speed > 0.05:
            boat_angle = np.arctan2(vy, vx)
        else:
            # Si arrêt, on prend l'angle vers le but (heuristic de départ)
            boat_angle = np.arctan2(gy - y, gx - x)

        wind_angle = np.arctan2(wy, wx)
        goal_angle = np.arctan2(gy - y, gx - x)

        # --- 1. Angle Relatif Vent (Point of Sail) ---
        # C'est LE paramètre physique. 
        # On mappe [-pi, pi] -> [0, wind_bins-1]
        rel_wind = self._angle_diff(wind_angle, boat_angle)
        wind_idx = int((rel_wind + np.pi) / (2 * np.pi) * self.wind_bins)
        wind_idx = np.clip(wind_idx, 0, self.wind_bins - 1)

        # --- 2. Angle Relatif But (Heading error) ---
        # On mappe [-pi, pi] -> [0, goal_bins-1]
        rel_goal = self._angle_diff(goal_angle, boat_angle)
        goal_idx = int((rel_goal + np.pi) / (2 * np.pi) * self.goal_bins)
        goal_idx = np.clip(goal_idx, 0, self.goal_bins - 1)

        # --- 3. Speed Bin ---
        if speed < 0.2: s_idx = 0
        elif speed < 1.2: s_idx = 1
        else: s_idx = 2

        # --- 4. Angular Velocity Bin (Stabilisation) ---
        # On approxime la rotation par le produit vectoriel ou l'inertie
        # Pour faire simple sans historique complexe :
        # On suppose que si le rudder est à fond, on tourne.
        # Ici on va ignorer ça pour l'état Q-table (trop grand state space)
        # et gérer la stabilité via la reward et l'action smoothing.
        
        return (wind_idx, goal_idx, s_idx)

    # ============================================================
    # 3. ACTION
    # ============================================================

    def act(self, obs, info=None):
        state = self.discretize_state(obs, info)

        # Initialisation "Optimiste" pour encourager l'exploration
        if state not in self.q_table:
            self.q_table[state] = np.full(9, 5.0)

        # Epsilon-Greedy
        if self.rng.random() < self.epsilon:
            action = self.rng.integers(0, 9)
        else:
            # On prend la meilleure action
            # Petit bruit aléatoire pour briser les égalités
            q_values = self.q_table[state]
            action = int(np.argmax(q_values + self.rng.normal(0, 0.01, size=9)))
            
        self.last_action = action
        return action

    # ============================================================
    # 4. LEARNING AVEC VMG REWARD
    # ============================================================

    def learn(self, obs, action, reward, next_obs, done, info=None):
        state = self.discretize_state(obs, info)
        next_state = self.discretize_state(next_obs, info)

        if state not in self.q_table: self.q_table[state] = np.full(9, 5.0)
        if next_state not in self.q_table: self.q_table[next_state] = np.full(9, 5.0)

        # --- Custom Reward Shaping (Le Secret) ---
        shaped_reward = self._compute_reward(obs, next_obs, done, info, action)

        # Q-Learning Update
        best_next_q = np.max(self.q_table[next_state])
        td_target = shaped_reward + self.gamma * best_next_q * (1 - float(done))
        current_q = self.q_table[state][action]
        
        self.q_table[state][action] = current_q + self.alpha * (td_target - current_q)

    def _compute_reward(self, obs, next_obs, done, info, action):
        """
        Calcule une reward basée purement sur la VMG et la physique.
        """
        # 1. Extraction Goal
        if info and "goal" in info:
            gx, gy = info["goal"]
        else:
            gx, gy = 16, 31

        # 2. VMG (Velocity Made Good)
        # Distance gagnée vers le but
        dist_prev = np.hypot(gx - obs[0], gy - obs[1])
        dist_next = np.hypot(gx - next_obs[0], gy - next_obs[1])
        vmg = (dist_prev - dist_next) 
        
        # Scaling factor pour que la VMG soit significative (~0.1 par step)
        # On multiplie par 200 => reward ~ +20 par step si on va vite vers le but
        reward = vmg * 200.0
        
        # 3. Penalité No-Go Zone (Si on est face au vent, on est mort)
        wx, wy = obs[4], obs[5]
        vx, vy = obs[2], obs[3]
        if np.hypot(vx, vy) > 0.01:
            boat_angle = np.arctan2(vy, vx)
            wind_angle = np.arctan2(wy, wx)
            angle_diff = abs(self._angle_diff(boat_angle, wind_angle))
            
            # Si on est trop près du vent (No-Go), grosse punition
            if angle_diff > (np.pi - self.no_go_angle):
                reward -= 10.0
        
        # 4. Step Penalty (Time is money)
        reward -= 0.5
        
        # 5. Bonus de Victoire
        if done:
            reward += 500.0
            
        return reward

    # ============================================================
    # 5. IO
    # ============================================================

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)

    def reset(self):
        self.last_action = 4

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)