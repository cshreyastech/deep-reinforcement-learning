import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from dueling_dqn import DuelingQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer
from prioritized_replay_buffer import PrioritizedReplayBuffer


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
IS_DQN = True
PER = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, seed, use_per=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size # (8)
        self.action_size = action_size # (4)
        self.seed = random.seed(seed)
        self.use_per = use_per
        
        # Q-Network
        # self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        # self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
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
                
                
#             if len(self.memory) > BATCH_SIZE:
#                 experiences = self.memory.sample()
#                 self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))        

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Implements Double DQN: use the local network to select the best next action,
        and the target network to evaluate its Q-value.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self.use_per:
            states, actions, rewards, next_states, dones, indices, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
            weights = torch.ones_like(rewards).to(device)  # uniform weights if not using PER
            indices = None
        
        # states, actions, rewards, next_states, dones = experiences

        Q_targets_next = None
        if IS_DQN:
            # Get max predicted Q values (for next states) from target model
            # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1) # (batch_size, 1)
            # .detach() is used so gradients don’t propagate into the target network
            # (and to avoid backprop through the next-state computations).
            next_target_q = self.qnetwork_target(next_states).detach() # (BATCH_SIZE, action_size)
            next_actions = next_target_q.argmax(dim=1).unsqueeze(1)  # shape: (batch_size, 1)
            Q_targets_next = next_target_q.gather(1, next_actions)  # shape: (batch_size, 1)
        else:
            # ------------------ Double DQN target calculation ------------------
            # Use local model to choose the best next action (action selection)        
            next_local_q = self.qnetwork_local(next_states).detach() # (BATCH_SIZE, action_size)
            next_actions = next_local_q.argmax(dim=1).unsqueeze(1)  # shape: (batch_size, 1)

            # Use target model to evaluate the chosen actions (action evaluation)
            next_target_q = self.qnetwork_target(next_states).detach()
            Q_targets_next = next_target_q.gather(1, next_actions)  # shape: (batch_size, 1)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # -----------------------------------------------------------------

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        
        # TD errors
        td_errors = Q_expected - Q_targets

        # Weighted loss
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities if using PER
        if self.use_per:
            self.memory.update_priorities(indices, td_errors.detach())

        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
        
        
        
        
        
        
        
        
        
        
        
#         -------------
#         # Compute loss
#         loss = F.mse_loss(Q_expected, Q_targets)
#         # Minimize the loss
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         # ------------------- update target network ------------------- #
#         self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

#     def learn(self, experiences, gamma):
#         states, actions, rewards, next_states, dones, indices, weights = experiences

#         # Double DQN targets
#         next_local_q = self.qnetwork_local(next_states).detach()
#         next_actions = next_local_q.argmax(dim=1).unsqueeze(1)
#         next_target_q = self.qnetwork_target(next_states).detach()
#         Q_targets_next = next_target_q.gather(1, next_actions)

#         Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
#         Q_expected = self.qnetwork_local(states).gather(1, actions)

#         # TD error
#         td_errors = Q_expected - Q_targets

#         # Loss with importance-sampling weights
#         loss = (weights * td_errors.pow(2)).mean()

#         # Optimize
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         # Update priorities
#         self.memory.update_priorities(indices, td_errors.detach())

#         # Soft update target network
#         self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)