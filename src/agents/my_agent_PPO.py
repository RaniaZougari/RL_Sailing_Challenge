import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.clip_eps = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        logits, value = self.model(state)
        probs = torch.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), value

    def evaluate(self, states, actions):
        logits, values = self.model(states)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values.squeeze()

    def update(self, trajectories):
        states, actions, rewards, dones, old_log_probs, values = trajectories

        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        old_log_probs = torch.tensor(old_log_probs)
        values = torch.tensor(values)

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):  # PPO epochs
            log_probs, entropy, new_values = self.evaluate(states, actions)
            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - new_values).pow(2).mean()

            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
