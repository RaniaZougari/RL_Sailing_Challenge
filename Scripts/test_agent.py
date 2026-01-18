from agents.my_agent import MyAgent
from env.env_sailing import SailingEnv
from env.wind_scenarios import get_wind_scenario


def test_agent(agent, wind_scenario_name, num_episodes=5, max_steps=1000):
    env = SailingEnv(**get_wind_scenario(wind_scenario_name))
    agent.exploration_rate = 0.0

    print("Testing trained agent...\n")

    for episode in range(num_episodes):
        observation, info = env.reset(seed=1000 + episode)
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(observation)
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if done or truncated:
                break

        print(
            f"Test Episode {episode+1}: "
            f"Steps={step+1}, Reward={total_reward}, "
            f"Position={info['position']}, Goal reached={done}"
        )


if __name__ == "__main__":
    agent = MyAgent()
    agent.load("models/my_agent.pkl")

    test_agent(agent, "training_1")
