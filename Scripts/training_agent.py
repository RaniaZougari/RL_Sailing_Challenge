import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

import numpy as np
import time

from agents.my_agent_physics import MyAgent
from wind_scenarios import get_wind_scenario
from env_sailing import SailingEnv

from matplotlib import pyplot as plt


def train_agent(
    agent,
    wind_scenarios,  # Can be a string or a list of strings
    num_episodes=1000,
    max_steps=500,
    save_path="models/trained_agent.pkl",
    seed=42,
):


    # Handle backward compatibility: convert string to list
    if isinstance(wind_scenarios, str):
        wind_scenarios = [wind_scenarios]
    
    # Reproducibility
    np.random.seed(seed)
    agent.seed(seed)

    # Global tracking
    all_rewards = []
    all_steps = []
    all_success = []
    scenario_metrics = {}

    
    print(f"Starting training on {len(wind_scenarios)} scenario(s) "
          f"for {num_episodes} total episodes...")
    print(f"Scenarios: {', '.join(wind_scenarios)}")
    print(f"Episodes per scenario: {num_episodes}\n")

    start_time = time.time()

    # Train on each scenario
    for scenario_idx, scenario_name in enumerate(wind_scenarios):
        print(f"\n{'='*60}")
        print(f"Training on scenario: {scenario_name} ({scenario_idx+1}/{len(wind_scenarios)})")
        print(f"{'='*60}")
        
        env = SailingEnv(**get_wind_scenario(scenario_name))
        
        scenario_rewards = []
        scenario_steps = []
        scenario_success = []

        for episode in range(num_episodes):
            observation, info = env.reset(seed=scenario_idx * num_episodes + episode)
            state = agent.discretize_state(observation)

            total_reward = 0

            for step in range(max_steps):
                action = agent.act(observation)

                next_observation, reward, done, truncated, info = env.step(action)
                next_state = agent.discretize_state(next_observation)

                agent.learn(state, action, reward, next_state)

                state = next_state
                observation = next_observation
                total_reward += reward

                if done or truncated:
                    break

            scenario_rewards.append(total_reward)
            scenario_steps.append(step + 1)
            scenario_success.append(done)

            if (episode + 1) % 10 == 0:
                success_rate = sum(scenario_success[-10:]) / 10 * 100
                print(
                    f"  Episode {episode+1}/{num_episodes} | "
                    f"Success rate (last 10): {success_rate:.1f}% | "
                    f"Epsilon: {agent.exploration_rate:.3f}"
                )

        # Store scenario metrics
        scenario_metrics[scenario_name] = {
            'success_rate': sum(scenario_success) / len(scenario_success) * 100,
            'avg_reward': np.mean(scenario_rewards),
            'avg_steps': np.mean(scenario_steps),
        }
        
        # Add to global metrics
        all_rewards.extend(scenario_rewards)
        all_steps.extend(scenario_steps)
        all_success.extend(scenario_success)

        print(f"\n  ✓ {scenario_name}: "
              f"Success={scenario_metrics[scenario_name]['success_rate']:.1f}%, "
              f"Reward={scenario_metrics[scenario_name]['avg_reward']:.1f}, "
              f"Steps={scenario_metrics[scenario_name]['avg_steps']:.1f}")

    training_time = time.time() - start_time

    # Save agent
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"\nTrained agent saved to '{save_path}'")

    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Time: {training_time:.1f}s")
    print(f"Overall Success rate: {sum(all_success)/len(all_success)*100:.1f}%")
    print(f"Overall Average reward: {np.mean(all_rewards):.2f}")
    print(f"Overall Average steps: {np.mean(all_steps):.1f}")
    print(f"Q-table size: {len(agent.q_table)}")
    
    print("\nPer-scenario performance:")
    for scenario_name, metrics in scenario_metrics.items():
        print(f"  {scenario_name}: "
              f"Success={metrics['success_rate']:.1f}%, "
              f"Reward={metrics['avg_reward']:.1f}, "
              f"Steps={metrics['avg_steps']:.1f}")

    return {
        "rewards": all_rewards,
        "steps": all_steps,
        "success": all_success,
        "scenario_metrics": scenario_metrics,
    }


def visualize_training_results(rewards_history, steps_history, success_history):
    # Calculate rolling averages
    window_size = 10
    rolling_rewards = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
    rolling_steps = np.convolve(steps_history, np.ones(window_size)/window_size, mode='valid')
    rolling_success = np.convolve([1 if s else 0 for s in success_history], np.ones(window_size)/window_size, mode='valid') * 100

    # Create the plots
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Plot rewards
    ax1.plot(rolling_rewards)
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training Progress (10-episode rolling average)')

    # Plot steps
    ax2.plot(rolling_steps)
    ax2.set_ylabel('Average Steps')

    # Plot success rate
    #ax3.plot(rolling_success)
    #ax3.set_ylabel('Success Rate (%)')
    #ax3.set_xlabel('Episode')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    agent = MyAgent()

    training = train_agent(
        agent=agent,
        wind_scenario_name="training_1",
        num_episodes=100,
        save_path="models/physics_agent.pkl",
    )
    visualize_training_results(rewards_history=training["rewards"],
        steps_history=training["steps"],
        success_history=training["success"],
    )
