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
    wind_scenario_name,
    num_episodes=1000,
    max_steps=1000,
    save_path="models/trained_agent.pkl",
    seed=42,
):


    # Reproducibility
    np.random.seed(seed)
    agent.seed(seed)

    env = SailingEnv(**get_wind_scenario(wind_scenario_name))

    rewards_history = []
    steps_history = []
    success_history = []

    print(f"Starting training on '{wind_scenario_name}' "
          f"for {num_episodes} episodes...")

    start_time = time.time()



    for episode in range(num_episodes):
        observation, info = env.reset(seed=episode)
        state = agent.discretize_state(observation)

        total_reward = 0

        for step in range(max_steps):
            action = agent.act(observation)

            next_observation, reward, done, truncated, info = env.step(action)
            next_state = agent.discretize_state(next_observation)

            agent.learn(
                state,
                action,
                reward,
                next_state
            )

            state = next_state
            observation = next_observation
            total_reward += reward

            if done or truncated:
                break


        rewards_history.append(total_reward)
        steps_history.append(step + 1)
        success_history.append(done)

        if (episode + 1) % 10 == 0:
            success_rate = sum(success_history[-10:]) / 10 * 100
            print(
                f"Episode {episode+1}/{num_episodes} | "
                f"Success rate (last 10): {success_rate:.1f}% | "
                f"Epsilon: {agent.exploration_rate:.3f}"
            )

    training_time = time.time() - start_time

    # Save agent
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"Trained agent saved to '{save_path}'")


    print("\nTraining completed!")
    print(f"Time: {training_time:.1f}s")
    print(f"Success rate: {sum(success_history)/len(success_history)*100:.1f}%")
    print(f"Average reward: {np.mean(rewards_history):.2f}")
    print(f"Average steps: {np.mean(steps_history):.1f}")
    print(f"Q-table size: {len(agent.q_table)}")

    return {
        "rewards": rewards_history,
        "steps": steps_history,
        "success": success_history,
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
