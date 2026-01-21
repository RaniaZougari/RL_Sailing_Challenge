"""
Greedy VMG Agent - Physics Based
Cet agent ne fait pas d'apprentissage (RL). Il utilise les lois physiques du jeu
pour choisir mathématiquement la direction qui le rapproche le plus vite du but à chaque instant.
"""

import numpy as np
# IMPORTANT : Sur Codabench, il faut importer depuis evaluator.base_agent
try:
    from evaluator.base_agent import BaseAgent
except ImportError:
    # Fallback pour le test local si nécessaire
    from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        # Position du goal (fixe selon le sujet)
        self.goal_position = np.array([16, 31])
        
    def _action_to_direction(self, action):
        """Convertit l'index d'action en vecteur direction."""
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1),
            (0, 0) # 8: Stay
        ]
        return np.array(directions[action])

    def calculate_efficiency_internal(self, boat_direction, wind_direction):
        """
        Copie locale de calculate_sailing_efficiency pour garantir
        l'indépendance du fichier sur Codabench.
        """
        # Invert wind direction to get where wind is coming FROM
        wind_from = -wind_direction
        
        # Calculate angle between wind and direction
        # Clip pour éviter les erreurs numériques hors de [-1, 1]
        dot_prod = np.clip(np.dot(wind_from, boat_direction), -1.0, 1.0)
        wind_angle = np.arccos(dot_prod)
        
        # Calculate sailing efficiency based on angle to wind
        if wind_angle < np.pi/4:  # < 45 degrees (No-Go Zone)
            sailing_efficiency = 0.05  
        elif wind_angle < np.pi/2:  # 45-90 degrees
            sailing_efficiency = 0.5 + 0.5 * (wind_angle - np.pi/4) / (np.pi/4)
        elif wind_angle < 3*np.pi/4:  # 90-135 degrees
            sailing_efficiency = 1.0
        else:  # > 135 degrees
            sailing_efficiency = 1.0 - 0.5 * (wind_angle - 3*np.pi/4) / (np.pi/4)
            sailing_efficiency = max(0.5, sailing_efficiency)
        
        return sailing_efficiency

    def act(self, observation, info=None):
        """
        Choisit l'action qui maximise la Vitesse Utile (VMG) vers la cible.
        """
        # 1. Récupérer les infos de l'observation
        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5] # Vent local
        
        current_pos = np.array([x, y])
        wind_vec = np.array([wx, wy])
        wind_mag = np.linalg.norm(wind_vec)
        
        # 2. Vecteur vers le but
        to_goal = self.goal_position - current_pos
        dist_goal = np.linalg.norm(to_goal)
        
        # Si on est très près, on s'arrête ou on continue (détail)
        if dist_goal < 0.5:
            return 8 # Stay
            
        to_goal_norm = to_goal / dist_goal

        # Si pas de vent, on ne peut rien faire de stratégique -> Stay ou direction but
        if wind_mag < 0.01:
            return 8 

        wind_dir_norm = wind_vec / wind_mag

        # 3. Évaluer chaque action possible (0 à 7)
        best_action = 8
        best_vmg = -np.inf

        for action in range(8):
            # Direction du bateau pour cette action
            boat_dir = self._action_to_direction(action)
            boat_dir_norm = boat_dir / np.linalg.norm(boat_dir)
            
            # A. Efficacité polaire (vitesse potentielle dans cette direction)
            efficiency = self.calculate_efficiency_internal(boat_dir_norm, wind_dir_norm)
            
            # B. Vitesse réelle estimée
            boat_speed = efficiency * wind_mag
            
            # C. VMG = Projection de la vitesse sur la direction du but
            # VMG = Vitesse * cos(angle_vers_but)
            # Produit scalaire de deux vecteurs unitaires = cos(angle)
            cos_to_goal = np.dot(boat_dir_norm, to_goal_norm)
            
            vmg = boat_speed * cos_to_goal
            
            # On prend la meilleure VMG
            if vmg > best_vmg:
                best_vmg = vmg
                best_action = action

        return best_action