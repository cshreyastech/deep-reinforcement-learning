import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from dqn_agent import Agent

if __name__ == '__main__':
  env_name = "LunarLander-v2"

  env = gym.make(env_name)
  # env.seed(0)
  observation, info = env.reset(seed=0)
  print('State shape: ', env.observation_space.shape)
  print('Number of actions: ', env.action_space.n)

  agent = Agent(state_size=8, action_size=4, seed=0)

  # # watch an untrained agent
  # state, probability = env.reset()
  # for j in range(200):
  #   action = agent.act(state)
  #   env.render()
  #   next_state, reward, done, truncated, info  = env.step(action)
  #   if done:
  #     break 

  """
  n_episodes (int): maximum number of training episodes
  max_t (int): maximum number of timesteps per episode
  eps_start (float): starting value of epsilon, for epsilon-greedy action selection
  eps_end (float): minimum value of epsilon
  eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
  """
  n_episodes = 2
  max_t      = 1000
  eps_start  = 1.0
  eps_end    = 0.01
  eps_decay  = 0.995


  scores = []                        # list containing scores from each episode
  scores_window = deque(maxlen=100)  # last 100 scores
  eps = eps_start                    # initialize epsilon
  for i_episode in range(1, n_episodes+1):
    state, probability = env.reset()
    score = 0
    for t in range(max_t):
      action = agent.act(state, eps)

      """
      next_state <class 'numpy.ndarray'>, (state_size)
      reward <class 'numpy.float64'>
      reward <class 'bool'>
      truncated <class 'bool'>
      info <class 'dict'>
      """
      next_state, reward, done, truncated, info = env.step(action)
      agent.step(state, action, reward, next_state, done)
      state = next_state
      score += reward
      if done:
        break 
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    eps = max(eps_end, eps_decay*eps) # decrease epsilon
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if i_episode % 100 == 0:
      print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    if np.mean(scores_window)>=200.0:
      print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
      torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
      break

  env.close()

  print()
  # plot the scores
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.plot(np.arange(len(scores)), scores)
  plt.ylabel('Score')
  plt.xlabel('Episode #')
  plt.show()
  plt.savefig("DQN_performance")

