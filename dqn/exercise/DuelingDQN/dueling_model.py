import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNetwork(nn.Module):
  """Actor (Policy) Model."""

  def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
    """Initialize parameters and build model.
    Params
    ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
    """
    super(DuelingQNetwork, self).__init__()
    self.seed = torch.manual_seed(seed)
    "*** YOUR CODE HERE ***"
    self.fc1 = nn.Linear(state_size, fc1_units)
    self.fc2 = nn.Linear(fc1_units, fc2_units)
    self.V = nn.Linear(fc2_units, 1)
    self.A = nn.Linear(fc2_units, action_size)
  def forward(self, state):
    """Build a network that maps state -> action values."""
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    V = self.V(x)
    A = self.A(x)
    
    result = torch.add(V, A - A.mean())
    # print("result", result.shape)
    return result

# self.duqnetwork_local DuelingQNetwork(
#   (fc1): Linear(in_features=8, out_features=64, bias=True)
#   (fc2): Linear(in_features=64, out_features=64, bias=True)
#   (V): Linear(in_features=64, out_features=1, bias=True)
#   (A): Linear(in_features=64, out_features=4, bias=True)
# )