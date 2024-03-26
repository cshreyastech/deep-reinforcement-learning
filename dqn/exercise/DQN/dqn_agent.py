import numpy as np
import random
# from collections import namedtuple, deque
import sys
sys.path.insert(1, '../')
from model import QNetwork

from replay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)

class Agent():
  """Interacts with and learns from the environment."""

  def __init__(self, state_size, action_size, seed):
    """Initialize an Agent object.
    
    Params
    ======
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        seed (int): random seed
    """
    self.state_size = state_size
    self.action_size = action_size
    self.seed = random.seed(seed)

    # Q-Network
    self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
    self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
    self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

    # Replay memory
    self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)
    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = 0
  
  def step(self, state, action, reward, next_state, done):
    # Save experience in replay memory
    self.memory.add(state, action, reward, next_state, done)
    
    # Learn every UPDATE_EVERY time steps.
    self.t_step = (self.t_step + 1) % UPDATE_EVERY
    if self.t_step == 0:
      # If enough samples are available in memory, get random subset and learn
      if len(self.memory) > BATCH_SIZE:
        experiences = self.memory.sample()
        self.learn(experiences, GAMMA)

  def act(self, state, eps=0.):
    """Returns actions for given state as per current policy.
    
    Params
    ======
      state (array_like): current state
      eps (float): epsilon, for epsilon-greedy action selection
    """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device) # (1 x state_size)
    self.qnetwork_local.eval()
    with torch.no_grad():
      action_values = self.qnetwork_local(state) # (1 x action_size)
    self.qnetwork_local.train()

    # Epsilon-greedy action selection
    if random.random() > eps:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  def learn(self, experiences, gamma):
    """Update value parameters using given batch of experience tuples.

    Params
    ======
      experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
      gamma (float): discount factor
    """
    states, actions, rewards, next_states, dones = experiences
    ## TODO: compute and minimize the loss
    "*** YOUR CODE HERE ***"
    # https://ai.stackexchange.com/questions/21515/is-there-any-good-reference-for-double-deep-q-learning
    # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms

    q_local_current = self.qnetwork_local(states) #(BATCH_SIZE x action_size)
    # q_expected = torch.gather(expected_local_rewards, 1, actions)
    q_expected = q_local_current.gather(1, actions) #(BATCH_SIZE x 1)
    #############

    q_target_next = self.qnetwork_target(next_states) #(BATCH_SIZE x action_space)

    # select the maximum reward for each of the next actions
    q_target_next_max = q_target_next.max(dim=1, keepdim=True)[0] #(BATCH_SIZE x 1)
    q_target = rewards + gamma * q_target_next_max * (1 - dones) #(BATCH_SIZE x 1)
    ##############

    loss = F.mse_loss(q_expected, q_target) # single tensor
    # Minimize the loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # ------------------- update target network ------------------- #
    self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

  def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
      local_model (PyTorch model): weights will be copied from
      target_model (PyTorch model): weights will be copied to
      tau (float): interpolation parameter 
    """
    # print("local_model:", local_model.parameters().data)
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      """
      target_param torch.Size([64, 8])
      target_param torch.Size([64])
      target_param torch.Size([64, 64])
      target_param torch.Size([64])
      target_param torch.Size([4, 64])
      target_param torch.Size([4])
      """
      target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
