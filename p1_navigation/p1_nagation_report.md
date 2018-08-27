# Value Based Deep Reinforcement Learning of Unity environment

## Project Details
Train an agent to navigate (and collect bananas!) in a large Unity, square world

## Environment Details
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Implementation Details
Initial the Unity agent was trained using vanilla DQN value-based implementation. Later, multiple improvements were implemented which includes Double DQN, Dueling DQN and Prioritized Experience Replay.


DuelingDQN had two 64 fully connected units (fc1, fc2). Third fully connected layer (fc3) has advantage output size set to number of actions and value output size set to one.

The output of fc1 was connected to separate advantage and value fc2 units. Outputs of both fc2 were fed to fc3 advantage and fc3 value. Both fc1 and fc2 had relu activators. value of fc3 was averaged across all state-actions. Finally, the mean of advantage and value was returned as final output. 

The agent was trained with headless implementaion in aws server.

## Results
The agent showed improvement in training episodes as the improvisations were implemented. The initial model took 1200 episodes, Double DQN took 550 episodes, combining Double and Dueling DQN showed an huge improvement to reach 337 episodes to reach average score of 13. Prioritized Experience Replay was implemented however the implementation could not find any valid state after 333 episodes. This would need further debugging.

## Future Improvements
* Debug and implement Prioritized Experience Replay
* Implement learning from pixels directly
