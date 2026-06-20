import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim):
        super().__init__()
        self.layer1 =nn.Linear(state_dim,hidden_dim)
        self.layer2 =nn.Linear(hidden_dim,hidden_dim)
        self.layer3 =nn.Linear(hidden_dim,action_dim)

        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        q_values = self.layer3(x)

        return q_values