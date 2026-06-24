import os
import numpy as np
import pickle
from dataclasses import dataclass

from .puyo import GAMEOVER_REWARD


@dataclass
class ReplayConfig:
    max_capacity: int = 1000
    normalize_returns: bool = False    # désactivé : incompatible en l'état avec l'échelle de node.value en MCTS
    value_target_mix: float = 0.5      # 0 = retour Monte-Carlo pur, 1 = valeur MCTS pure

    def __post_init__(self):
        assert 0.0 <= self.value_target_mix <= 1.0, "value_target_mix doit être dans [0, 1]"


class EpisodeBuffer:
    """
    Class for storing a single episode (one whole puyo game)
    """

    def __init__(self, discount_factor):
        self.observations = []
        self.rewards = []
        self.returns = []
        self.policies = []
        self.values = []

        self.discount_factor = discount_factor

    def store_transition(self, observation, reward, policy, value):
        self.observations.append(observation)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)

    def compute_returns(self):
        """
        computes the discounted returns for the whole game
        """
        value = 0
        for i in reversed(range(len(self.rewards))):
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
        episode.compute_returns()

        mix = self.config.value_target_mix
        blended_targets = [
            (1 - mix) * mc_return + mix * mcts_value
            for mc_return, mcts_value in zip(episode.returns, episode.values)
        ]

        self.observations.extend(episode.observations)
        self.returns.extend(blended_targets)
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
        current_size = len(self.observations)
        if current_size < batch_size:
            batch_size = current_size
        indices = np.random.choice(range(current_size), size=batch_size, replace=False)

        batch_observations = np.array(self.observations)[indices, :, :, :]
        all_returns = np.array(self.returns)
        batch_returns = all_returns[indices]
        batch_policies = np.array(self.policies)[indices, :]

        if self.config.normalize_returns:
            mean = all_returns.mean()
            std = all_returns.std() + 1e-8
            batch_returns = (batch_returns - mean) / std

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