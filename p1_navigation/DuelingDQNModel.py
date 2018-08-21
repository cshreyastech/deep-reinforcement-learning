import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingDQN, self).__init__()
        self.action_size = action_size
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        
        self.fc2_advantage = nn.Linear(fc1_units, fc2_units)
        self.fc2_value = nn.Linear(fc1_units, fc2_units)
        
        self.fc3_advantage = nn.Linear(fc2_units, action_size)
        self.fc3_value = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # return self.fc3(x)
    
        x = F.relu(self.fc1(state))
        
        advantage = F.relu(self.fc2_advantage(x))
        value = F.relu(self.fc2_value(x))
        
        advantage = self.fc3_advantage(advantage)
        value = self.fc3_value(value).expand(x.size(0), self.action_size)
        
        x = value + advantage - advantage.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        return x