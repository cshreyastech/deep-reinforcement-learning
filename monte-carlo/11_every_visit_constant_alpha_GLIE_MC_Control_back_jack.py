# Algorithm 9: Every-Visit MC Prediction (for action values)
import sys
import gymnasium as gym
import numpy as np
from collections import defaultdict

from plot_utils import plot_blackjack_values, plot_policy

# from warnings import filterwarnings
# filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


def get_probs(Q_s, epsilon, nA):
  """ obtains the action probabilities corresponding to epsilon-greedy policy """
  policy_s = np.ones(nA) * epsilon / nA
  best_a = np.argmax(Q_s)
  policy_s[best_a] = 1 - epsilon + (epsilon / nA)
  return policy_s

def generate_episode_from_Q(env, Q, epsilon, nA):
  episode = []
  # state = bj_env.reset()
  state, info = env.reset()
  while True:
    action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                                if state in Q else env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    episode.append((state, action, reward))
    state = next_state
    if done:
      break
  return episode

def update_Q(env, episode, Q, alpha, gamma):
  states, actions, rewards = zip(*episode)
  # prepare for discounting
  discounts = np.array([gamma**i for i in range(len(rewards)+1)])
  # update the sum of the returns, number of visits, and action-value 
  # function estimates for each state-action pair in the episode
  for i, state in enumerate(states):
    old_Q = Q[state][actions[i]] 
    G = sum(rewards[i:]*discounts[:-(1+i)])
    Q[state][actions[i]] = old_Q + alpha*(G - old_Q)
  return Q

def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
  nA = env.action_space.n
  # initialize empty dictionary of arrays
  Q = defaultdict(lambda: np.zeros(nA))
  epsilon = eps_start
  # loop over episodes
  for i_episode in range(1, num_episodes+1):
    # monitor progress
    if i_episode % 1000 == 0:
      print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
      sys.stdout.flush()
    
    ## TODO: complete the function
    epsilon = max(epsilon * eps_decay, eps_min)

    episode = generate_episode_from_Q(env, Q, epsilon, nA)
    # update the action-value function estimate using the episode
    Q = update_Q(env, episode, Q, alpha, gamma)
      
  # determine the policy corresponding to the final action-value function estimate
  policy = dict((k, np.argmax(v)) for k, v in Q.items())
  return policy, Q

if __name__ == '__main__':
  env = gym.make('Blackjack-v1')
  # print(env.observation_space)
  # print(env.action_space)
  # 500000
  # obtain the estimated optimal policy and action-value function
  policy, Q = mc_control(env, 500_000, 0.02)  # obtain the corresponding state-value function

  V_to_plot = dict((k,(k[0]>18)*(np.dot([0.8, 0.2],v)) + (k[0]<=18)*(np.dot([0.2, 0.8],v))) \
    for k, v in Q.items())

  # plot the state-value function
  plot_blackjack_values(V_to_plot)

  print()