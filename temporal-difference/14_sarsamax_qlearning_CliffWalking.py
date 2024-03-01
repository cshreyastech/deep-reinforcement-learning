import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import wrappers
import pickle # save models
# https://gymnasium.farama.org/environments/toy_text/cliff_walking/

def max_action(Q, state, actions):
  values = np.array([Q[state, a] for a in actions])
  action = np.argmax(values)

  return action

if __name__ == '__main__':
  env = gym.make('CliffWalking-v0')
  n_games = 500
  alpha = 0.1
  gamma = 0.99
  eps = 0

  nS = env.observation_space.n
  nA = env.action_space.n


  states = []
  for s in range(nS):
    states.append((s))

  actions = []
  for a in range(nA):
    actions.append(a)

  load = False

  modle_name_pkl = 'acrobot_sarsamax_qlearning.pkl'
  if load == False:
    Q = {}
    for state in states:
      for action in actions:
        Q[state, action] = 0
  else:
    pickle_in = open(modle_name_pkl, 'rb')
    Q = pickle.load(pickle_in)
    # env = wrappers.Monitor(env, "tmp/acrobot", video_callable=lambda episode_id: True, force=True)


  score = 0
  total_reward = np.zeros(n_games)

  for i in range(n_games):
    state, probability = env.reset()

    done = False

    if i % 100 == 0:
      print('episode ', i, ' score ', score, ' eps ', eps)

    score = 0
    while not done:
      action = max_action(Q, state, actions) if np.random.random() > eps else env.action_space.sample()
    
      observation_ = env.step(action)
      state_, reward, done, truncated, info = observation_

      action_ = max_action(Q, state_, actions) #if np.random.random() > eps else env.action_space.sample()

      score += reward
      Q[state, action] += alpha * (reward + gamma * Q[state_, action_] - Q[state, action])

      state = state_

    total_reward[i] = score
    eps = eps - 2 / n_games if eps > 0.01 else 0.01
  
  mean_rewards = np.zeros(n_games)
  for t in range(n_games):
    mean_rewards[t] = np.mean(total_reward[max(0, t-50):(t+1)])
  plt.plot(mean_rewards)
  plt.show()

  # f = open(modle_name_pkl, 'wb')
  # pickle.dump(Q, f)
  # f.close()