import os
import numpy as np
import pickle
from dataclasses import dataclass

from .puyo import GAMEOVER_REWARD


@dataclass
class ReplayConfig:
    """
    Dataclass for configuring the replay buffer
    """
    max_capacity: int = 1000

    def __post_init__(self):
        pass


class EpisodeBuffer:
    """
    Class for storing a single episode (one whole puyo game)
    """
    def __init__(self, discount_factor):
        self.observations = []
        self.rewards = []
        self.returns = []
        self.policies = []

        self.discount_factor = discount_factor

    def store_transition(self, observation, reward, policy):
        """
        store a single transition
        """
        self.observations.append(observation)
        self.rewards.append(reward)
        self.policies.append(policy)

    def store_terminal_state(self, observation, terminal_reward):
        """
        store the terminal state of the game (used for storing game over moves)
        """
        self.observations.append(observation)
        self.rewards.append(terminal_reward)
        self.policies.append(np.zeros((22, )))

    def compute_returns(self):
        """
        computes the discounted returns for the whole game
        """
        value = 0

        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1 and self.rewards[i] == GAMEOVER_REWARD:
                value = self.rewards[i]
            else:
                value = self.rewards[i] + self.discount_factor * value

            self.returns.insert(0, value)


class ReplayBuffer:
    """
    Class for the Replay Buffer
    """
    def __init__(self, name, config=ReplayConfig()):
        self.name = name
        self.config = config

        self.observations = []
        self.returns = []
        self.policies = []

    def add_episode(self, episode):
        """
        add a single episode to the buffer
        """
        episode.compute_returns()
        self.observations.extend(episode.observations)
        self.returns.extend(episode.returns)
        self.policies.extend(episode.policies)

    def trim_buffer(self):
        """
        ensures the replay buffer does not go over maximum capacity
        """
        current_size = len(self.observations)
        if current_size > self.config.max_capacity:
            n_removed = current_size - self.config.max_capacity
            self.observations = self.observations[n_removed:]
            self.returns = self.returns[n_removed:]
            self.policies = self.policies[n_removed:]

    def sample_batch(self, batch_size):
        """
        sample a random batch of training examples
        """
        current_size = len(self.observations)
        if current_size < batch_size:
            batch_size = current_size
        indices = np.random.choice(range(current_size), size=batch_size, replace=False)

        batch_observations = np.array(self.observations)[indices, :, :, :]
        batch_returns = np.array(self.returns)[indices]
        batch_policies = np.array(self.policies)[indices, :]

        return batch_observations, batch_returns, batch_policies

    def save(self, path_to_dir):
        """
        save replay buffer data to pickle file
        """
        with open(os.path.join(path_to_dir, self.name + '_replay_buffer.pkl'), 'wb') as f1:
            pickle.dump((self.observations, self.returns, self.policies), f1)

    def load(self, path_to_dir):
        """
        load replay buffer data from pickle file
        """
        with open(os.path.join(path_to_dir, self.name + '_replay_buffer.pkl'), 'rb') as f1:
            self.observations, self.returns, self.policies = pickle.load(f1)
