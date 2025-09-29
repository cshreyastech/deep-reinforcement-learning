import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
    """Dueling DQN Network: state -> Q-values."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Shared layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # Value stream
        self.value_fc = nn.Linear(fc2_units, fc2_units)
        self.value = nn.Linear(fc2_units, 1)

        # Advantage stream
        self.adv_fc = nn.Linear(fc2_units, fc2_units)
        self.advantage = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Forward pass: state -> Q-values"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Value stream
        v = F.relu(self.value_fc(x))
        v = self.value(v)

        # Advantage stream
        a = F.relu(self.adv_fc(x))
        a = self.advantage(a)

        # Combine streams
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q
