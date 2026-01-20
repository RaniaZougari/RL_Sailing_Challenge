#!/usr/bin/env python3
"""
Analyse de Corrélation: Scores Locaux vs Score Professeur

Ce script implémente plusieurs stratégies pour prédire le score du professeur
à partir des évaluations sur 5 scénarios locaux.

Stratégies implémentées:
1. Analyse de corrélation par scénario
2. Régression Ridge pour prédiction
3. Leave-One-Out Cross-Validation
4. Score composite (moyenne + pénalité de variance)
5. Visualisations et recommandations
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# Configuration des graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Scénarios locaux utilisés pour l'évaluation
SCENARIOS = ['simple_static', 'static_headwind', 'training_1', 'training_2', 'training_3']


class ScoreAnalyzer:
    """Analyseur de corrélation entre scores locaux et scores du professeur."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.models_data = []
        self.X = None  # Matrice des scores locaux
        self.y = None  # Vecteur des scores du prof
        self.model_names = []
        self.scaler = StandardScaler()
        
    def extract_prof_score_from_filename(self, filename: str) -> Optional[float]:
        """Extrait le score du prof du nom de fichier."""
        # Pattern: *_X_points.json ou *_X.Y_points.json
        match = re.search(r'_([0-9]+(?:\.[0-9]+)?)_points\.json$', filename)
        if match:
            return float(match.group(1))
        return None
    
    def load_data(self) -> int:
        """Charge tous les fichiers avec scores du prof."""
        json_files = list(self.results_dir.glob("*points.json"))
        
        for json_file in json_files:
            prof_score = self.extract_prof_score_from_filename(json_file.name)
            if prof_score is None:
                continue
                
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extraire les scores locaux
                if 'evaluation' not in data:
                    continue
                    
                local_scores = []
                missing_scenario = False
                for scenario in SCENARIOS:
                    if scenario not in data['evaluation']:
                        missing_scenario = True
                        break
                    score = data['evaluation'][scenario].get('custom_score', 0)
                    local_scores.append(score)
                
                if missing_scenario:
                    print(f"⚠️  {json_file.name}: scénarios manquants, ignoré")
                    continue
                
                self.models_data.append({
                    'filename': json_file.name,
                    'prof_score': prof_score,
                    'local_scores': local_scores,
                    'agent': data.get('metadata', {}).get('agent', 'unknown')
                })
                
            except Exception as e:
                print(f"❌ Erreur lors du chargement de {json_file.name}: {e}")
        
        if self.models_data:
            # Trier par score du prof
            self.models_data.sort(key=lambda x: x['prof_score'])
            
            # Créer matrices numpy
            self.X = np.array([m['local_scores'] for m in self.models_data])
            self.y = np.array([m['prof_score'] for m in self.models_data])
            self.model_names = [m['agent'] for m in self.models_data]
        
        return len(self.models_data)
    
    def print_data_summary(self):
        """Affiche un résumé des données chargées."""
        print("\n" + "="*80)
        print("📊 DONNÉES CHARGÉES")
        print("="*80)
        print(f"\nNombre de modèles: {len(self.models_data)}")
        print(f"Scénarios évalués: {len(SCENARIOS)}")
        print(f"\nScores du professeur: {sorted([m['prof_score'] for m in self.models_data])}")
        
        print(f"\n{'Agent':<40} {'Prof':<10} {'Scores Locaux (moyenne)':<25}")
        print("-"*80)
        for model in self.models_data:
            avg_local = np.mean(model['local_scores'])
            print(f"{model['agent']:<40} {model['prof_score']:<10.2f} {avg_local:<25.4f}")
    
    def analyze_correlations(self) -> Dict[str, float]:
        """Calcule les corrélations entre chaque scénario et le score prof."""
        correlations = {}
        
        print("\n" + "="*80)
        print("🔍 ANALYSE DE CORRÉLATION PAR SCÉNARIO")
        print("="*80)
        print(f"\n{'Scénario':<20} {'Corrélation':<15} {'Interprétation':<30}")
        print("-"*80)
        
        for i, scenario in enumerate(SCENARIOS):
            corr = np.corrcoef(self.X[:, i], self.y)[0, 1]
            correlations[scenario] = corr
            
            if abs(corr) > 0.7:
                interpretation = "✅ Très prédictif"
            elif abs(corr) > 0.4:
                interpretation = "⚠️  Modérément prédictif"
            else:
                interpretation = "❌ Peu prédictif"
            
            print(f"{scenario:<20} {corr:>8.4f}      {interpretation:<30}")
        
        return correlations
    
    def train_ridge_regression(self, alpha: float = 1.0) -> Ridge:
        """Entraîne un modèle de régression Ridge."""
        # Normaliser les features
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Entraîner le modèle
        model = Ridge(alpha=alpha)
        model.fit(X_scaled, self.y)
        
        return model
    
    def analyze_feature_importance(self, model: Ridge):
        """Analyse l'importance de chaque scénario dans le modèle."""
        print("\n" + "="*80)
        print("📈 IMPORTANCE DES SCÉNARIOS (Coefficients Ridge)")
        print("="*80)
        
        coefficients = model.coef_
        
        # Créer un dataframe pour trier
        importance = [(SCENARIOS[i], abs(coefficients[i])) for i in range(len(SCENARIOS))]
        importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{'Scénario':<20} {'Coefficient':<15} {'Importance Absolue':<20}")
        print("-"*80)
        for scenario, coef_abs in importance:
            idx = SCENARIOS.index(scenario)
            coef = coefficients[idx]
            print(f"{scenario:<20} {coef:>10.4f}      {coef_abs:>10.4f}")
    
    def leave_one_out_cross_validation(self, alpha: float = 1.0) -> Dict[str, float]:
        """Effectue une validation croisée Leave-One-Out."""
        n = len(self.models_data)
        predictions = np.zeros(n)
        
        print("\n" + "="*80)
        print("🔄 LEAVE-ONE-OUT CROSS-VALIDATION")
        print("="*80)
        print(f"\n{'Modèle':<40} {'Vrai':<10} {'Prédit':<10} {'Erreur':<10}")
        print("-"*80)
        
        for i in range(n):
            # Créer train/test sets
            train_idx = [j for j in range(n) if j != i]
            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            X_test = self.X[i:i+1]
            
            # Normaliser
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Entraîner et prédire
            model = Ridge(alpha=alpha)
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)[0]
            
            predictions[i] = pred
            error = abs(pred - self.y[i])
            
            print(f"{self.model_names[i]:<40} {self.y[i]:<10.2f} {pred:<10.2f} {error:<10.2f}")
        
        # Calculer les métriques
        mae = mean_absolute_error(self.y, predictions)
        rmse = np.sqrt(mean_squared_error(self.y, predictions))
        r2 = r2_score(self.y, predictions)
        
        print("\n" + "-"*80)
        print(f"MAE (Erreur Absolue Moyenne):  {mae:.4f}")
        print(f"RMSE (Racine Erreur Carrée):   {rmse:.4f}")
        print(f"R² Score:                       {r2:.4f}")
        
        return {
            'predictions': predictions,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def calculate_composite_score(self, alpha: float = 0.7, beta: float = 0.3) -> np.ndarray:
        """Calcule un score composite basé sur moyenne et variance."""
        means = np.mean(self.X, axis=1)
        stds = np.std(self.X, axis=1)
        
        # Normaliser pour que std soit sur la même échelle
        if stds.max() > 0:
            stds_normalized = stds / stds.max() * means.max()
        else:
            stds_normalized = stds
        
        composite = alpha * means - beta * stds_normalized
        
        print("\n" + "="*80)
        print(f"🎯 SCORE COMPOSITE (α={alpha}, β={beta})")
        print("="*80)
        print("\nFormule: score = α × moyenne(scores_locaux) - β × std(scores_locaux)")
        print("Favorise les modèles performants ET stables\n")
        
        print(f"{'Modèle':<40} {'Composite':<12} {'Moyenne':<12} {'Std':<12}")
        print("-"*80)
        for i, model in enumerate(self.models_data):
            print(f"{self.model_names[i]:<40} {composite[i]:<12.4f} {means[i]:<12.4f} {stds[i]:<12.4f}")
        
        return composite
    
    def create_visualizations(self, output_dir: str = "results/correlation_analysis"):
        """Crée toutes les visualisations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Heatmap de corrélation
        self._plot_correlation_heatmap(output_path)
        
        # 2. Barplot des corrélations
        self._plot_correlation_barplot(output_path)
        
        # 3. Scatter plots par scénario
        self._plot_scenario_scatters(output_path)
        
        # 4. Prédictions vs Réalité (LOOCV)
        self._plot_predictions_vs_actual(output_path)
        
        print(f"\n✅ Visualisations sauvegardées dans: {output_path}")
    
    def _plot_correlation_heatmap(self, output_path: Path):
        """Heatmap des corrélations."""
        # Créer matrice de corrélation complète (scénarios + score prof)
        data_with_prof = np.column_stack([self.X, self.y])
        corr_matrix = np.corrcoef(data_with_prof.T)
        
        labels = SCENARIOS + ['Score Prof']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                    xticklabels=labels, yticklabels=labels, 
                    center=0, vmin=-1, vmax=1)
        plt.title('Matrice de Corrélation: Scénarios Locaux vs Score Prof', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_heatmap.png', dpi=300)
        plt.close()
    
    def _plot_correlation_barplot(self, output_path: Path):
        """Barplot des corrélations par scénario."""
        correlations = {}
        for i, scenario in enumerate(SCENARIOS):
            correlations[scenario] = np.corrcoef(self.X[:, i], self.y)[0, 1]
        
        plt.figure(figsize=(12, 6))
        scenarios = list(correlations.keys())
        corr_values = list(correlations.values())
        colors = ['green' if c > 0.7 else 'orange' if c > 0.4 else 'red' for c in [abs(c) for c in corr_values]]
        
        bars = plt.bar(scenarios, corr_values, color=colors, alpha=0.7, edgecolor='black')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Très prédictif (|r| > 0.7)')
        plt.axhline(y=-0.7, color='green', linestyle='--', alpha=0.5)
        plt.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Modérément prédictif (|r| > 0.4)')
        plt.axhline(y=-0.4, color='orange', linestyle='--', alpha=0.5)
        
        plt.xlabel('Scénario', fontsize=12)
        plt.ylabel('Corrélation avec Score Prof', fontsize=12)
        plt.title('Corrélation de chaque scénario avec le Score du Professeur', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_barplot.png', dpi=300)
        plt.close()
    
    def _plot_scenario_scatters(self, output_path: Path):
        """Scatter plots pour chaque scénario."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, scenario in enumerate(SCENARIOS):
            ax = axes[i]
            x = self.X[:, i]
            corr = np.corrcoef(x, self.y)[0, 1]
            
            ax.scatter(x, self.y, s=100, alpha=0.6, edgecolors='black')
            
            # Ligne de tendance
            z = np.polyfit(x, self.y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel(f'Score Local ({scenario})', fontsize=10)
            ax.set_ylabel('Score Prof', fontsize=10)
            ax.set_title(f'{scenario}\nCorr = {corr:.3f}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Cacher le dernier subplot (5 scénarios sur grille 2x3)
        axes[5].axis('off')
        
        plt.suptitle('Relation entre Scores Locaux et Score Prof', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'scenario_scatters.png', dpi=300)
        plt.close()
    
    def _plot_predictions_vs_actual(self, output_path: Path):
        """Plot des prédictions LOOCV vs valeurs réelles."""
        loocv_results = self.leave_one_out_cross_validation()
        predictions = loocv_results['predictions']
        
        plt.figure(figsize=(10, 10))
        plt.scatter(self.y, predictions, s=150, alpha=0.6, edgecolors='black')
        
        # Ligne parfaite
        min_val = min(self.y.min(), predictions.min())
        max_val = max(self.y.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction Parfaite')
        
        # Annoter chaque point
        for i, name in enumerate(self.model_names):
            plt.annotate(f'{name}\n({self.y[i]:.1f})', 
                        (self.y[i], predictions[i]), 
                        fontsize=8, alpha=0.7,
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Score Prof (Réel)', fontsize=12)
        plt.ylabel('Score Prédit (LOOCV)', fontsize=12)
        plt.title(f'Prédictions vs Réalité (Leave-One-Out Cross-Validation)\n' + 
                 f'MAE = {loocv_results["mae"]:.2f}, R² = {loocv_results["r2"]:.3f}',
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'predictions_vs_actual.png', dpi=300)
        plt.close()
    
    def predict_for_new_model(self, json_path: str, ridge_model: Ridge) -> Dict:
        """Prédit le score prof pour un nouveau modèle."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        local_scores = []
        for scenario in SCENARIOS:
            if scenario not in data.get('evaluation', {}):
                raise ValueError(f"Scénario {scenario} manquant dans {json_path}")
            score = data['evaluation'][scenario].get('custom_score', 0)
            local_scores.append(score)
        
        X_new = np.array([local_scores])
        X_new_scaled = self.scaler.transform(X_new)
        
        prediction = ridge_model.predict(X_new_scaled)[0]
        mean_score = np.mean(local_scores)
        std_score = np.std(local_scores)
        
        # Score composite
        composite = 0.7 * mean_score - 0.3 * std_score
        
        return {
            'filename': Path(json_path).name,
            'local_scores': local_scores,
            'mean_local': mean_score,
            'std_local': std_score,
            'ridge_prediction': prediction,
            'composite_score': composite,
            'final_estimate': 0.7 * prediction + 0.3 * mean_score
        }
    
    def generate_recommendations(self, ridge_model: Ridge):
        """Génère des recommandations pour les futures soumissions."""
        print("\n" + "="*80)
        print("💡 RECOMMANDATIONS POUR FUTURES SOUMISSIONS")
        print("="*80)
        
        # 1. Meilleur modèle actuel
        best_idx = np.argmax(self.y)
        print(f"\n✅ Meilleur modèle actuel: {self.model_names[best_idx]}")
        print(f"   Score prof: {self.y[best_idx]:.2f}")
        print(f"   Scores locaux: {self.X[best_idx]}")
        
        # 2. Scénarios les plus importants
        coeffs = ridge_model.coef_
        importance = [(SCENARIOS[i], abs(coeffs[i])) for i in range(len(SCENARIOS))]
        importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 Prioriser ces scénarios (ordre d'importance):")
        for i, (scenario, _) in enumerate(importance[:3], 1):
            print(f"   {i}. {scenario}")
        
        # 3. Seuil de soumission
        best_score = self.y[best_idx]
        threshold = best_score * 1.05  # 5% mieux
        
        print(f"\n⚠️  Seuil de soumission recommandé:")
        print(f"   Ne soumettre que si score_estimé > {threshold:.2f}")
        print(f"   (soit {best_score:.2f} + 5% de marge)")


def main():
    parser = argparse.ArgumentParser(description="Analyse de corrélation scores locaux vs prof")
    parser.add_argument('--results-dir', default='results', help='Répertoire des résultats')
    parser.add_argument('--check-data', action='store_true', help='Vérifier les données uniquement')
    parser.add_argument('--full-analysis', action='store_true', help='Analyse complète avec visualisations')
    parser.add_argument('--predict', type=str, help='Prédire le score pour un fichier JSON')
    parser.add_argument('--output-dir', default='results/correlation_analysis', help='Dossier de sortie')
    parser.add_argument('--alpha', type=float, default=1.0, help='Paramètre de régularisation Ridge')
    
    args = parser.parse_args()
    
    # Initialiser l'analyseur
    analyzer = ScoreAnalyzer(args.results_dir)
    
    # Charger les données
    n_models = analyzer.load_data()
    
    if n_models == 0:
        print("❌ Aucun modèle avec score prof trouvé!")
        return
    
    print(f"\n✅ {n_models} modèles chargés avec succès!")
    
    if args.check_data:
        analyzer.print_data_summary()
        return
    
    # Analyse complète
    analyzer.print_data_summary()
    
    # 1. Corrélations
    correlations = analyzer.analyze_correlations()
    
    # 2. Régression Ridge
    ridge_model = analyzer.train_ridge_regression(alpha=args.alpha)
    analyzer.analyze_feature_importance(ridge_model)
    
    # 3. LOOCV
    loocv_results = analyzer.leave_one_out_cross_validation(alpha=args.alpha)
    
    # 4. Score composite
    composite_scores = analyzer.calculate_composite_score()
    
    # 5. Recommandations
    analyzer.generate_recommendations(ridge_model)
    
    # 6. Visualisations
    if args.full_analysis:
        analyzer.create_visualizations(args.output_dir)
    
    # 7. Prédiction pour nouveau modèle
    if args.predict:
        print("\n" + "="*80)
        print("🔮 PRÉDICTION POUR NOUVEAU MODÈLE")
        print("="*80)
        result = analyzer.predict_for_new_model(args.predict, ridge_model)
        print(f"\nFichier: {result['filename']}")
        print(f"Scores locaux: {result['local_scores']}")
        print(f"Moyenne locale: {result['mean_local']:.4f}")
        print(f"Écart-type: {result['std_local']:.4f}")
        print(f"Prédiction Ridge: {result['ridge_prediction']:.2f}")
        print(f"Score composite: {result['composite_score']:.4f}")
        print(f"\n🎯 ESTIMATION FINALE: {result['final_estimate']:.2f}")


if __name__ == "__main__":
    main()
