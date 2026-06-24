from itertools import count
from pathlib import Path

import gymnasium as gym
import torch

from common.logger import Logger
from src.agents.dqn_agent import DQNAgent
from src.common.config import Config
from src.common.device import get_device


def train(config):
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = get_device()

    agent = DQNAgent(
        state_dim,
        action_dim,
        config,
        device,
    )

    logger = Logger()

    for episode in range(config.num_episodes):
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
            next_state = torch.tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
            agent.store_transition(state, action, reward_tensor, next_state, done)

            metrics = agent.update()
            if metrics:
                logger.log(metrics)

            state = next_state
            episode_reward += reward
            if done:
                break

        logger.log({"episode_reward": episode_reward})
        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode} | "
                f"Reward={logger.mean('episode_reward', 10):.1f} | "
                f"Loss={logger.mean('loss', 100):.4f} | "
                f"Epsilon={logger.metrics['epsilon'][-1]:.3f}"
            )

    env.close()

    return agent


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    config = Config.from_yaml(project_root / "configs" / "dqn_cartpole.yaml")
    print(config)
    agent = train(config)
    save_path = project_root / "models" / "dqn_cartpole" / "dqn_cartpole.pth"
    agent.save(save_path)
    print(f"Model saved to: {save_path}")
