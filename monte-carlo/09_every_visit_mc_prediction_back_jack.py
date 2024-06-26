# Algorithm 9: Every-Visit MC Prediction (for action values)
import sys
import gym
import numpy as np
from collections import defaultdict

from plot_utils import plot_blackjack_values, plot_policy

# from warnings import filterwarnings
# filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


def generate_episode_from_limit_stochastic(bj_env):
  episode = []
  # state = bj_env.reset()
  state, info = env.reset()
  while True:
    probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
    action = np.random.choice(np.arange(2), p=probs)
    next_state, reward, done, truncated, info = bj_env.step(action)
    episode.append((state, action, reward))
    state = next_state
    if done:
      break
  return episode

def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
  # initialize empty dictionaries of arrays
  returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
  N = defaultdict(lambda: np.zeros(env.action_space.n))
  Q = defaultdict(lambda: np.zeros(env.action_space.n))
  # loop over episodes
  for i_episode in range(1, num_episodes+1):
    # monitor progress
    if i_episode % 1000 == 0:
      print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
      sys.stdout.flush()
    
    ## TODO: complete the function
    # generate an episode
    episode = generate_episode(env)
    # obtain states, actions, rewards
    states, actions, rewards = zip(*episode)
    
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    # update the sum of the returns, number of visits, and action-value 
    # function estimates for each state-action pair in the episode
    for i, state in enumerate(states):
      G = sum(rewards[i:]*discounts[:-(1+i)])
      returns_sum[state][actions[i]] += G #sum(rewards[i:]*discounts[:-(1+i)])
      N[state][actions[i]] += 1
      Q[state][actions[i]] = returns_sum[state][actions[i]] / N[state][actions[i]]
  return Q

if __name__ == '__main__':
  env = gym.make('Blackjack-v1')
  # print(env.observation_space)
  # print(env.action_space)
  # 500000
  Q = mc_prediction_q(env, 50, generate_episode_from_limit_stochastic)
  # obtain the corresponding state-value function
  V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
    for k, v in Q.items())

  # plot the state-value function
  plot_blackjack_values(V_to_plot)

  print()