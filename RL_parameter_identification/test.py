from __future__ import annotations

import gym_examples
from gymnasium.wrappers import FlattenObservation
from gym_examples.wrappers import RelativePosition
import gymnasium as gym


import numpy as np
from collections import defaultdict
#from typing import Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

#env = gym.make('gym_examples/GridWorld-v0', render_mode="human")
env = gym.make('gym_examples/GridWorld-v0')
#wrapped_env = FlattenObservation(env)
env = RelativePosition(env)

# reset the environment to get the first observation
done = False
observation, info = env.reset()


class Agent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int], explore: True) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        if explore == True:
            # with probability epsilon return a random action to explore the environment
            if np.random.random() < self.epsilon:
                return env.action_space.sample()

            # with probability (1 - epsilon) act greedily (exploit)
            else:
                obs = tuple(obs)
                return int(np.argmax(self.q_values[obs]))
        else:
            obs = tuple(obs)
            return int(np.argmax(self.q_values[obs]))
            

    def update(
        self,
        obs: tuple[int, int],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int],
    ):
        """Updates the Q-value of an action."""
        obs = tuple(obs)
        next_obs = tuple( next_obs)
        #print(self.q_values[next_obs])
      #  print(self.q_values)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
       # print(self.q_values)
        
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )
        
       # print(self.q_values[obs][action]) 
         
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)




# training


#print(wrapped_env.reset())     # E.g.  [3 0 3 3], {}
# hyperparameters

learning_rate = 0.01
#n_episodes = 100_000
n_episodes = 1000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = Agent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)


env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        
        action = agent.get_action(obs,True)
        
        #print(action)
        
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()


'''
for _ in range(100):
    
    action = env.action_space.sample()  # agent policy that uses the observation and info
    
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
'''


# %%
# Visualizing the training
# ------------------------------
#

rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()


# plot test cases

env = gym.make('gym_examples/GridWorld-v0', render_mode="human")

#env = gym.make('GridWorld-v0', render_mode="human")

env = RelativePosition(env)
done = False
observation, info = env.reset()

for _ in range(100):
    
    action = agent.get_action(obs,False)  # agent policy that uses the observation and info
    
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()