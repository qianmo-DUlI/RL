from itertools import count
from pathlib import Path

import gymnasium as gym
import torch

from src.agents.dqn_agent import DQNAgent
from src.common.config import Config
from src.common.device import get_device
from utils import Logger
from utils import plot_training_curves
from utils.checkpoint import load_checkpoint, save_checkpoint


def setup_dirs(project_root):
    checkpoint_dir = project_root / "models" / "dqn_cartpole" / "checkpoints"
    log_dir = project_root / "results" / "logs" / "dqn_cartpole"
    best_model_dir = project_root / "models" / "dqn_cartpole" / "best_model.pth"
    return checkpoint_dir, log_dir, best_model_dir


def train(config, project_root, checkpoint_dir, log_dir, best_model_dir):
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
    start_episode = 0
    resume_path = checkpoint_dir / "latest.pth"
    if resume_path.exists():
        start_episode = load_checkpoint(resume_path, agent, device) + 1
        print(f"Resume training from episode {start_episode}")
    logger = Logger(log_dir=log_dir)
    best_reward = float("-inf")
    for episode in range(start_episode, config.num_episodes):
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
            if agent.total_steps % 1000 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{agent.total_steps}.pth"
                save_checkpoint(checkpoint_path, agent, episode)
                save_checkpoint(checkpoint_dir / "latest.pth", agent, episode)
            if metrics:
                logger.log(metrics, step=agent.total_steps)

            state = next_state
            episode_reward += reward
            if done:
                break

        logger.log({"episode_reward": episode_reward}, step=episode + 1)
        if (episode + 1) % 100 == 0:
            reward_last_100 = logger.mean("episode_reward", window=100)
            if reward_last_100 > best_reward:
                best_reward = reward_last_100
                agent.save(best_model_dir)
                print(
                    f"New Best Model! "
                    f"Avg Reward last 100 episodes={reward_last_100:.2f}"
                )
        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1} | "
                f"Reward={logger.mean('episode_reward', 10):.1f} | "
                f"Loss={logger.mean('loss', 100):.4f} | "
                f"Epsilon={logger.latest('epsilon'):.3f}"
            )

    logger.close()
    env.close()
    plot_training_curves(logger, project_root / "results" / "figures")
    return agent


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    config = Config.from_yaml(project_root / "configs" / "dqn_cartpole.yaml")
    checkpoint_dir, log_dir, best_model_dir = setup_dirs(project_root)

    print(config)
    agent = train(config, project_root, checkpoint_dir, log_dir, best_model_dir)
    save_path = project_root / "models" / "dqn_cartpole" / "dqn_cartpole.pth"
    agent.save(save_path)
    print(f"Model saved to: {save_path}")
