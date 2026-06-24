import torch


def save_checkpoint(path, agent, episode):
    checkpoint = {"agent": agent.state_dict(), "episode": episode}
    torch.save(checkpoint, path)


def load_checkpoint(path, agent, device):
    checkpoint = torch.load(path, map_location=device)
    agent.load_state_dict(checkpoint["agent"])
    return checkpoint["episode"]
