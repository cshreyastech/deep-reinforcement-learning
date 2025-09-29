import random
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer with proportional prioritization."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Params
        ======
            action_size (int): action dimension
            buffer_size (int): max buffer size
            batch_size (int): batch size
            seed (int): random seed
            alpha (float): prioritization exponent (0 = uniform, 1 = full priority)
            beta_start (float): initial beta for importance-sampling correction
            beta_frames (int): annealing steps for beta
        """
        self.action_size = action_size
        self.memory = []
        self.max_size = buffer_size
        self.batch_size = batch_size
        self.pos = 0
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.seed = random.seed(seed)

        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience with max priority so it gets sampled at least once."""
        e = self.experience(state, action, reward, next_state, done)

        max_prio = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.max_size:
            self.memory.append(e)
        else:
            self.memory[self.pos] = e

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.max_size

    def sample(self):
        """Sample experiences with probability proportional to priorities."""
        if len(self.memory) == self.max_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        experiences = [self.memory[idx] for idx in indices]

        # Compute importance-sampling weights
        total = len(self.memory)
        beta = self.beta_by_frame()
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize

        # Convert to tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(weights).float().unsqueeze(1).to(device)

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, td_errors, epsilon=1e-5):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error.item()) + epsilon

    def beta_by_frame(self):
        """Anneal beta over frames."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def __len__(self):
        return len(self.memory)