import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

class PrioritizedReplayBuffer:
  """Fixed-size buffer to store experience tuples."""

  def __init__(self, action_size, buffer_size, batch_size, seed, device, priority_scale=0.7):
    """Initialize a ReplayBuffer object.

    Params
    ======
      action_size (int): dimension of each action
      buffer_size (int): maximum size of buffer
      batch_size (int): size of each training batch
      seed (int): random seed
    """
    self.action_size = action_size
    self.memory = deque(maxlen=buffer_size)
    self.priorities = deque(maxlen=buffer_size)
    self.batch_size = batch_size
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    self.seed = random.seed(seed)
    self.device = device
    self.priority_scale = priority_scale
    self.sample_indices = []
  
  def add(self, state, action, reward, next_state, done):
    """Add a new experience to memory."""
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)
    self.priorities.append(max(self.priorities, default=1))
  
  def set_priorities(self, errors, offset=0.1):
    for i, e in zip(self.sample_indices, errors):
      self.priorities[i] = abs(e) + offset

  def get_probabilities(self):
    scaled_priorities = np.array(self.priorities) ** self.priority_scale
    sample_probabilities = scaled_priorities / sum(scaled_priorities)
    return sample_probabilities

  def get_importance(self, probabilities):
    importance = 1 / len(self.memory) * 1 / probabilities
    importance_normalized = importance / max(importance)
    return importance_normalized

  def sample(self):
    """Randomly sample a batch of experiences from memory."""

    sample_size = min(len(self.memory), self.batch_size)
    sample_probabilities = self.get_probabilities()
    self.sample_indices = random.choices(range(len(self.memory)), k=sample_size, weights=sample_probabilities)
    # experiences = np.array(self.memory, dtype=object)[self.sample_indices]

    experiences = [self.memory[i] for i in self.sample_indices]

    importance = self.get_importance(sample_probabilities[self.sample_indices])


    states      = \
      torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
    
    # (BATCH_SIZE x 1)
    actions     = \
      torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)

    # (BATCH_SIZE x 1)
    rewards     = \
      torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)

    # (BATCH_SIZE x state_size)
    next_states = \
      torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)

    # (BATCH_SIZE x 1)
    dones       = \
      torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

    # importance
    importance_t = torch.from_numpy(importance).to(self.device)
    return (states, actions, rewards, next_states, dones, importance_t)

  def __len__(self):
    """Return the current size of internal memory."""
    return len(self.memory)