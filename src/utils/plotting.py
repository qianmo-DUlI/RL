from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(logger, save_dir):
    save_dir = Path(save_dir)
    metrics_to_plot = [
        "episode_reward",
        "loss",
        "epsilon",
        "avg_q",
        "avg_target_q",
    ]
    for metrics_name in metrics_to_plot:
        if metrics_name not in logger.metrics:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(logger.metrics[metrics_name])
        plt.title(metrics_name)
        if metrics_name == "episode_reward":
            plt.xlabel("Episode")
            plt.ylabel("Reward")
        if metrics_name == "loss":
            plt.xlabel("Update Step")
            plt.ylabel("Loss")
        if metrics_name == "epsilon":
            plt.xlabel("Step")
            plt.ylabel("Epsilon")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / f"{metrics_name}.png")
        plt.close()
