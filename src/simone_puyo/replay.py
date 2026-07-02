import os
import numpy as np
import pickle
from dataclasses import dataclass


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
    Class for the Replay Buffer.
    Stockage en tableaux numpy pre-alloues (buffer circulaire) : plus de
    reconstruction complete a partir de listes Python a chaque sample_batch().
    """
    def __init__(self, name, config=ReplayConfig()):
        self.name = name
        self.config = config

        self._observations = None   # alloues paresseusement au premier add_episode
        self._returns = None
        self._policies = None

        self._write_idx = 0   # prochaine position d'ecriture dans le buffer circulaire
        self._size = 0        # nombre d'entrees valides actuellement (<= max_capacity)

    def __len__(self):
        return self._size

    def _allocate(self, observation_shape, n_actions):
        capacity = self.config.max_capacity
        self._observations = np.zeros((capacity,) + observation_shape, dtype=np.float32)
        self._returns = np.zeros(capacity, dtype=np.float32)
        self._policies = np.zeros((capacity, n_actions), dtype=np.float32)

    def add_episode(self, episode):
        """
        ajoute un episode complet au buffer circulaire (ecriture vectorisee,
        avec gestion du wrap-around si l'episode chevauche la fin du tableau)
        """
        episode.compute_returns()

        observations = np.asarray(episode.observations, dtype=np.float32)
        policies = np.asarray(episode.policies, dtype=np.float32)
        returns = np.asarray(episode.returns, dtype=np.float32)
        values = np.asarray(episode.values, dtype=np.float32)

        mix = self.config.value_target_mix
        blended_targets = (1 - mix) * returns + mix * values

        n_new = len(observations)
        if n_new == 0:
            return

        if self._observations is None:
            self._allocate(observations.shape[1:], policies.shape[1])

        capacity = self.config.max_capacity
        if n_new > capacity:
            # cas degenere : un seul episode plus grand que toute la capacite
            observations = observations[-capacity:]
            policies = policies[-capacity:]
            blended_targets = blended_targets[-capacity:]
            n_new = capacity

        end_idx = self._write_idx + n_new
        if end_idx <= capacity:
            self._observations[self._write_idx:end_idx] = observations
            self._returns[self._write_idx:end_idx] = blended_targets
            self._policies[self._write_idx:end_idx] = policies
        else:
            first_part = capacity - self._write_idx
            second_part = n_new - first_part

            self._observations[self._write_idx:] = observations[:first_part]
            self._returns[self._write_idx:] = blended_targets[:first_part]
            self._policies[self._write_idx:] = policies[:first_part]

            self._observations[:second_part] = observations[first_part:]
            self._returns[:second_part] = blended_targets[first_part:]
            self._policies[:second_part] = policies[first_part:]

        self._write_idx = end_idx % capacity
        self._size = min(self._size + n_new, capacity)

    def sample_batch(self, batch_size):
        if self._size == 0:
            raise ValueError("Le replay buffer est vide.")

        batch_size = min(batch_size, self._size)
        indices = np.random.choice(self._size, size=batch_size, replace=False)

        batch_observations = self._observations[indices]
        batch_returns = self._returns[indices]
        batch_policies = self._policies[indices]

        if self.config.normalize_returns:
            valid_returns = self._returns[:self._size]
            mean = valid_returns.mean()
            std = valid_returns.std() + 1e-8
            batch_returns = (batch_returns - mean) / std

        return batch_observations, batch_returns, batch_policies

    def save(self, path_to_dir):
        with open(os.path.join(path_to_dir, self.name + '_replay_buffer.pkl'), 'wb') as f1:
            pickle.dump(
                (self._observations, self._returns, self._policies, self._write_idx, self._size),
                f1
            )

    def load(self, path_to_dir):
        with open(os.path.join(path_to_dir, self.name + '_replay_buffer.pkl'), 'rb') as f1:
            self._observations, self._returns, self._policies, self._write_idx, self._size = pickle.load(f1)