from itertools import count
from pathlib import Path

import gymnasium as gym
import torch

from src.agents.dqn_agent import DQNAgent
from src.common.config import Config


def evaluate(agent, env, device, num_episodes=10):
    agent.eval()

    rewards = []

    for _ in range(num_episodes):

        state, _ = env.reset()

        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        episode_reward = 0

        for _ in count():

            action = agent.select_action(state)

            obs, reward, terminated, truncated, _ = env.step(action.item())

            done = terminated or truncated

            state = torch.tensor(
                obs,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            episode_reward += reward

            if done:
                break

        rewards.append(episode_reward)

    avg_reward = sum(rewards) / len(rewards)

    print(f"Average Reward: {avg_reward:.2f}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    config = Config.from_yaml(project_root / "configs" / "dqn_cartpole.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(
        "CartPole-v1",
        render_mode="human",
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim,
        action_dim,
        config,
        device,
    )

    agent.load(project_root / "models" / "dqn_cartpole.pth")

    evaluate(
        agent,
        env,
        device,
        num_episodes=10,
    )

    env.close()
