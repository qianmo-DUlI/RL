from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from src.agents.dqn_agent import DQNAgent
from src.common.config import Config


def train(config: Config):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim, action_dim, config, device)

    for episode in range(config.num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # epsilon-greedy
            epsilon = max(
                config.epsilon_end,
                config.epsilon_start - (agent.total_steps / config.epsilon_decay),
            )

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = agent.policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, next_state, reward)
            agent.total_steps += 1

            state = next_state
            episode_reward += reward

            if len(agent.memory) >= config.batch_size:
                agent.update()

        print(
            f"Episode {episode:4d} | Reward: {episode_reward:6.1f} | Epsilon: {epsilon:.3f}"
        )

    env.close()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    config = Config.from_yaml(project_root / "configs" / "dqn_cartpole.yaml")
    print(config)
    # train(config)
