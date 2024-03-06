import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import wrappers
import pickle # save models
from collections import defaultdict, deque
import random
# https://gymnasium.farama.org/environments/toy_text/cliff_walking/

def epsilon_greedy(Q, state, nA, eps):
  """Selects epsilon-greedy action for supplied state.
  
  Params
  ======
      Q (dictionary): action-value function
      state (int): current state
      nA (int): number actions in the environment
      eps (float): epsilon
  """
  if random.random() > eps: # select greedy action with probability epsilon
      return np.argmax(Q[state])
  else:                     # otherwise, select an action randomly
      return random.choice(np.arange(env.action_space.n))

if __name__ == '__main__':
  env = gym.make('CliffWalking-v0')
  n_games = 5000
  alpha = 0.1
  gamma = 0.99
  eps = 0

  nS = env.observation_space.n
  nA = env.action_space.n



  # load = False

  # modle_name_pkl = 'acrobot_sarsa.pkl'
  # if load == False:

  # else:
  #   pickle_in = open(modle_name_pkl, 'rb')
  #   Q = pickle.load(pickle_in)
  #   # env = wrappers.Monitor(env, "tmp/acrobot", video_callable=lambda episode_id: True, force=True)


  Q = defaultdict(lambda: np.zeros(nA))  # initialize empty dictionary of arrays
  score = 0
  total_reward = np.zeros(n_games)

  for i in range(n_games):
    state, probability = env.reset()

    done = False

    if i % 100 == 0:
      print('episode ', i, ' score ', score, ' eps ', eps)

    score = 0

    action = epsilon_greedy(Q, state, nA, eps)            # epsilon-greedy action selection


    while not done:
      observation_ = env.step(action)
      state_, reward, done, truncated, info = observation_

      action_ = epsilon_greedy(Q, state_, nA, eps)

      score += reward

      Q[state][action] += alpha * (reward + gamma * Q[state_][action_] - Q[state][action])

      state = state_
      action = action_

    total_reward[i] = score
    eps = eps - 2 / n_games if eps > 0.01 else 0.01
  
  mean_rewards = np.zeros(n_games)
  for t in range(n_games):
    mean_rewards[t] = np.mean(total_reward[max(0, t-50):(t+1)])
  plt.plot(mean_rewards)
  plt.show()

  # # f = open(modle_name_pkl, 'wb')
  # # pickle.dump(Q, f)
  # # f.close()