# import sys
# import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.append(project_root)
# sys.path.append(os.path.join(project_root, 'src'))

# import numpy as np
# import time

# from agents.my_agent_DQ import MyAgent
# from wind_scenarios import get_wind_scenario
# from env_sailing import SailingEnv

# from matplotlib import pyplot as plt


# def train_agent(
#     agent,
#     wind_scenario_name,
#     num_episodes=1000,
#     max_steps=1000,
#     save_path="models/trained_agent.pkl",
#     seed=42,
# ):


#     # Reproducibility
#     np.random.seed(seed)
#     agent.seed(seed)

#     # Au début de train_agent
#     agent.epsilon = 1.0 
#     min_epsilon = 0.01
#     decay_rate = 0.998 # Décroissance lente au début, puis ça se stabilise

#     env = SailingEnv(**get_wind_scenario(wind_scenario_name))

#     rewards_history = []
#     steps_history = []
#     success_history = []

#     print(f"Starting training on '{wind_scenario_name}' "
#           f"for {num_episodes} episodes...")

#     start_time = time.time()



#     for episode in range(num_episodes):
#         observation, info = env.reset(seed=episode)
#         state = agent.discretize_state(observation)

#         total_reward = 0

#         for step in range(max_steps):
#             action = agent.act(observation)

#             next_observation, reward, done, truncated, info = env.step(action)

#             # 🔥 IMPORTANT : on passe LES OBSERVATIONS, PAS LES ÉTATS
#             agent.learn(
#                 observation,
#                 action,
#                 reward,
#                 next_observation,
#                 done
#             )

#             observation = next_observation
#             total_reward += reward

#             if done or truncated:
#                 break
#             agent.epsilon = max(min_epsilon, agent.epsilon * decay_rate)

#         rewards_history.append(total_reward)
#         steps_history.append(step + 1)
#         success_history.append(done)

#         agent.exploration_rate = max(0.05, agent.exploration_rate * 0.98)

#         if (episode + 1) % 10 == 0:
#             success_rate = sum(success_history[-10:]) / 10 * 100
#             print(
#                 f"Episode {episode+1}/{num_episodes} | "
#                 f"Success rate (last 10): {success_rate:.1f}% | "
#                 f"Epsilon: {agent.exploration_rate:.3f}"
#             )

#     training_time = time.time() - start_time

#     # Save agent
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     agent.save(save_path)
#     print(f"Trained agent saved to '{save_path}'")


#     print("\nTraining completed!")
#     print(f"Time: {training_time:.1f}s")
#     print(f"Success rate: {sum(success_history)/len(success_history)*100:.1f}%")
#     print(f"Average reward: {np.mean(rewards_history):.2f}")
#     print(f"Average steps: {np.mean(steps_history):.1f}")
#     print(f"Q-table size: {len(agent.q_table)}")

#     return {
#         "rewards": rewards_history,
#         "steps": steps_history,
#         "success": success_history,
#     }


# def visualize_training_results(rewards_history, steps_history, success_history):
#     # Calculate rolling averages
#     window_size = 10
#     rolling_rewards = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
#     rolling_steps = np.convolve(steps_history, np.ones(window_size)/window_size, mode='valid')
#     rolling_success = np.convolve([1 if s else 0 for s in success_history], np.ones(window_size)/window_size, mode='valid') * 100

#     # Create the plots
#     # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

#     # Plot rewards
#     ax1.plot(rolling_rewards)
#     ax1.set_ylabel('Average Reward')
#     ax1.set_title('Training Progress (10-episode rolling average)')

#     # Plot steps
#     ax2.plot(rolling_steps)
#     ax2.set_ylabel('Average Steps')

#     # Plot success rate
#     #ax3.plot(rolling_success)
#     #ax3.set_ylabel('Success Rate (%)')
#     #ax3.set_xlabel('Episode')

#     plt.tight_layout()
#     plt.show()



# if __name__ == "__main__":
#     agent = MyAgent()

#     training = train_agent(
#         agent=agent,
#         wind_scenario_name="training_1",
#         num_episodes=1000,   # increase later
#         save_path="models/trainedDQ_gem_1.pkl",
#     )
#     visualize_training_results(rewards_history=training["rewards"],
#         steps_history=training["steps"],
#         success_history=training["success"],
#     )

import sys
import os
import numpy as np
import time
from matplotlib import pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from agents.my_agent_DQ import MyAgent
from wind_scenarios import get_wind_scenario
from env_sailing import SailingEnv

def train_agent(
    agent,
    wind_scenario_name,
    num_episodes=1000,
    max_steps=1000,
    save_path="models/trained_agent.pkl",
    seed=42,
    start_epsilon=None # Optionnel, pour ne pas écraser l'epsilon actuel
):
    # Reproducibility
    np.random.seed(seed)
    agent.seed(seed)

    # Gestion de l'epsilon (exploration)
    if start_epsilon is not None:
        agent.epsilon = start_epsilon
    
    # Paramètres de décroissance
    min_epsilon = 0.05
    # Decay calculé pour atteindre min_epsilon vers 80% du training
    decay_rate = (min_epsilon / agent.epsilon) ** (1 / (num_episodes * 0.8)) if agent.epsilon > min_epsilon else 1.0

    env = SailingEnv(**get_wind_scenario(wind_scenario_name))

    rewards_history = []
    steps_history = []
    success_history = []

    print(f"Starting training on '{wind_scenario_name}' for {num_episodes} episodes...")
    print(f"Initial Epsilon: {agent.epsilon:.4f}")

    start_time = time.time()

    for episode in range(num_episodes):
        observation, info = env.reset(seed=episode)
        
        # Pas besoin de discretiser ici pour l'agent, il le fait dans act/learn
        # Mais pour le debug on peut garder ça simple
        
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(observation) # act gère la discrétisation

            next_observation, reward, done, truncated, info = env.step(action)

            # Apprentissage
            agent.learn(
                observation,
                action,
                reward,
                next_observation,
                done,
                info # On passe info pour que l'agent connaisse le goal précis
            )

            observation = next_observation
            total_reward += reward

            if done or truncated:
                break
        
        # --- FIN DE L'EPISODE ---
        
        # Décroissance de l'epsilon PAR EPISODE (pas par step)
        agent.epsilon = max(min_epsilon, agent.epsilon * decay_rate)

        rewards_history.append(total_reward)
        steps_history.append(step + 1)
        success_history.append(done)

        if (episode + 1) % 50 == 0:
            avg_rew = np.mean(rewards_history[-50:])
            avg_steps = np.mean(steps_history[-50:])
            success_rate = sum(success_history[-50:]) / 50 * 100
            print(
                f"Ep {episode+1}/{num_episodes} | "
                f"R: {avg_rew:.1f} | S: {avg_steps:.1f} | "
                f"Succ: {success_rate:.0f}% | "
                f"Eps: {agent.epsilon:.3f}"
            )

    training_time = time.time() - start_time

    # Save agent
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"Agent saved to '{save_path}'")

    return {
        "rewards": rewards_history,
        "steps": steps_history,
        "success": success_history,
    }

def train_robust_agent():
    """Entraîne l'agent séquentiellement sur plusieurs scénarios."""
    agent = MyAgent()
    
    scenarios = ["training_1", "training_2", "training_3"]
    
    # Pour stocker tout l'historique concaténé
    global_history = {
        "rewards": [],
        "steps": [],
        "success": []
    }
    
    final_save_path = "models/trainedDQ_FINAL2.pkl"

    for i, sc in enumerate(scenarios):
        print(f"\n==========================================")
        print(f">>> PHASE {i+1}/3 : SCENARIO {sc} <<<")
        print(f"==========================================")
        
        # Gestion intelligente de l'epsilon entre les scénarios
        # Au début on explore beaucoup (1.0).
        # Au scénario suivant, on garde un peu d'exploration (0.3) pour s'adapter au nouveau vent
        # mais on ne repart pas de zéro pour ne pas oublier ce qu'on sait faire.
        eps_start = 1.0 if i == 0 else 0.3
        
        # On entraîne
        results = train_agent(
            agent=agent,
            wind_scenario_name=sc,
            num_episodes=2000, # 2000 épisodes par vent pour bien bétonner
            save_path=final_save_path,
            start_epsilon=eps_start
        )
        
        # On ajoute les résultats à l'historique global
        global_history["rewards"].extend(results["rewards"])
        global_history["steps"].extend(results["steps"])
        global_history["success"].extend(results["success"])
            
    print("\n>>> ROBUST TRAINING FINISHED <<<")
    return global_history

def visualize_training_results(history):
    rewards = history["rewards"]
    steps = history["steps"]
    
    window_size = 50 # Lissage plus large vu qu'on a beaucoup d'épisodes
    
    rolling_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    rolling_steps = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax1.plot(rolling_rewards, color='blue', label='Reward')
    ax1.set_ylabel('Avg Reward')
    ax1.set_title(f'Global Training Progress (Moving Avg {window_size})')
    ax1.grid(True)

    ax2.plot(rolling_steps, color='orange', label='Steps')
    ax2.set_ylabel('Avg Steps')
    ax2.set_xlabel('Episodes (Cumulative)')
    ax2.grid(True)
    
    # Lignes verticales pour marquer les changements de scénarios (approx toutes les 2000 ep)
    for x in [2000, 4000]:
        ax1.axvline(x=x, color='red', linestyle='--', alpha=0.5)
        ax2.axvline(x=x, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Lancement du training robuste
    training_data = train_robust_agent()
    
    # Visualisation
    visualize_training_results(training_data)