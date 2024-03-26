import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class DQN(nn.Module):
  def __init__(self, state_size, action_size):
    super(DQN, self).__init__()
    self.fc1 = nn.Linear(state_size, 24)
    self.fc2 = nn.Linear(24, 24)
    self.fc3 = nn.Linear(24, action_size)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x






class PrioritizedReplayBuffer:
  def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
    self.capacity = capacity
    self.alpha = alpha
    self.beta = beta
    self.beta_increment_per_sampling = beta_increment_per_sampling
    self.buffer = []
    self.priorities = np.zeros((capacity,), dtype=np.float32)
    self.pos = 0
    self.size = 0

  def add(self, experience):
    max_priority = np.max(self.priorities) if self.size > 0 else 1.0
    if len(self.buffer) < self.capacity:
      self.buffer.append(experience)
    else:
      self.buffer[self.pos] = experience
    self.priorities[self.pos] = max_priority
    self.pos = (self.pos + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  def sample(self, batch_size):
    if self.size == self.capacity:
      priorities = self.priorities
    else:
      priorities = self.priorities[:self.size]
    probs = priorities ** self.alpha
    probs /= probs.sum()

    indices = np.random.choice(self.size, batch_size, p=probs)
    samples = [self.buffer[idx] for idx in indices]

    total = len(self.buffer)
    weights = (total * probs[indices]) ** (-self.beta)
    weights /= weights.max()

    self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

    return indices, samples, torch.FloatTensor(weights), probs

  def update_priorities(self, indices, errors):
    for idx, error in zip(indices, errors):
      self.priorities[idx] = error.item() + 1e-5

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
buffer_size = 10000
batch_size = 32
gamma = 0.99
learning_rate = 0.001
target_update = 10
num_episodes = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = PrioritizedReplayBuffer(buffer_size)

def compute_td_loss(batch):
  states, actions, rewards, next_states, dones = zip(*batch)
  states = torch.tensor(states, dtype=torch.float32).to(device)
  actions = torch.tensor(actions, dtype=torch.long).to(device)
  rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
  next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
  dones = torch.tensor(dones, dtype=torch.float32).to(device)

  state_action_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
  next_state_values = target_net(next_states).max(1)[0].detach()
  expected_state_action_values = rewards + (gamma * next_state_values * (1 - dones))

  loss = (state_action_values - expected_state_action_values).pow(2)
  weighted_loss = loss * weights
  td_errors = loss.detach().cpu().numpy()
  memory.update_priorities(indices, td_errors)
  return weighted_loss.mean()

def update_target_model():
  target_net.load_state_dict(policy_net.state_dict())

for ep in range(num_episodes):
  state,probability = env.reset()
  done = False
  total_reward = 0
  while not done:
    action = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)).argmax().item()
    next_state, reward, done, truncated, info = env.step(action)
    total_reward += reward
    memory.add(Transition(state, action, next_state, reward, done))
    state = next_state

    if len(memory.buffer) > batch_size:
      indices, batch, weights, probs = memory.sample(batch_size)
      loss = compute_td_loss(batch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  if ep % target_update == 0:
      update_target_model()

  print(f"Episode: {ep}, Total Reward: {total_reward}")
