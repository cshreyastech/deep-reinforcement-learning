from agent import Agent
from monitor import interact
import gymnasium as gym
import numpy as np

env = gym.make('Taxi-v3')
agent = Agent(env.action_space.n)

num_episodes = 20000
avg_rewards, best_avg_reward = interact(env, agent, num_episodes)