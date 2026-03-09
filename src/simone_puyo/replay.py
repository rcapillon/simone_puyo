import os
import numpy as np
import pickle
from dataclasses import dataclass

from .puyo import GAMEOVER_REWARD


@dataclass
class ReplayConfig:
    max_capacity: int = 100000

    def __post_init__(self):
        pass


class EpisodeBuffer:
    def __init__(self, discount_factor):
        self.observations = []
        self.rewards = []
        self.returns = []
        self.policies = []
        self.discount_factor = discount_factor

    def store_transition(self, observation, reward, policy):
        self.observations.append(observation)
        self.rewards.append(reward)
        self.policies.append(policy)

    def store_terminal_state(self, observation):
        """
        Stocke l'état terminal en cas de game over uniquement.
        reward = 0 : la pénalité GAMEOVER_REWARD est déjà dans le
        store_transition précédent (via game.step).
        Fournit au value head un exemple explicite : cet état vaut 0.
        Ne pas appeler en cas de fin par max_moves (valeur résiduelle non nulle).
        """
        self.observations.append(observation)
        self.rewards.append(0.)
        self.policies.append(np.zeros((22,)))

    def compute_returns(self):
        """
        Calcule les retours cumulés actualisés pour tout l'épisode.
        Pas de cas spécial nécessaire : reward de l'état terminal = 0,
        GAMEOVER_REWARD est déjà dans la transition précédente.
        """
        value = 0
        for i in reversed(range(len(self.rewards))):
            value = self.rewards[i] + self.discount_factor * value
            self.returns.insert(0, value)


class ReplayBuffer:
    def __init__(self, name, config=ReplayConfig()):
        self.name = name
        self.config = config
        self.observations = []
        self.returns = []
        self.policies = []

        # Statistiques calculées sur l'ensemble du buffer.
        # Plus stables qu'une normalisation par batch car elles
        # évoluent lentement au fil de l'entraînement.
        self._returns_mean = 0.
        self._returns_std = 1.

    def _update_normalization_stats(self):
        """
        Recalcule mean et std sur tous les returns du buffer.
        Appelé après chaque modification (add_episode, trim_buffer, load).
        """
        if len(self.returns) < 2:
            return
        arr = np.array(self.returns)
        self._returns_mean = float(arr.mean())
        self._returns_std = float(arr.std()) + 1e-8

    def add_episode(self, episode):
        episode.compute_returns()
        self.observations.extend(episode.observations)
        self.returns.extend(episode.returns)
        self.policies.extend(episode.policies)
        self._update_normalization_stats()

    def trim_buffer(self):
        current_size = len(self.observations)
        if current_size > self.config.max_capacity:
            n_removed = current_size - self.config.max_capacity
            self.observations = self.observations[n_removed:]
            self.returns = self.returns[n_removed:]
            self.policies = self.policies[n_removed:]
            self._update_normalization_stats()

    def sample_batch(self, batch_size):
        current_size = len(self.observations)
        if current_size < batch_size:
            batch_size = current_size
        indices = np.random.choice(range(current_size), size=batch_size, replace=False)

        batch_observations = np.array(self.observations)[indices, :, :, :]
        batch_returns = np.array(self.returns)[indices]
        batch_policies = np.array(self.policies)[indices, :]

        # Normalisation sur les stats du buffer complet.
        # Le value head apprend à prédire la valeur relative à la
        # distribution courante des retours → gradients plus stables.
        # L'UCT compare les Q-values entre elles, pas contre un seuil
        # absolu → le changement d'échelle ne dégrade pas la sélection.
        batch_returns = (batch_returns - self._returns_mean) / self._returns_std

        return batch_observations, batch_returns, batch_policies

    def save(self, path_to_dir):
        with open(os.path.join(path_to_dir, self.name + '_replay_buffer.pkl'), 'wb') as f1:
            pickle.dump((self.observations, self.returns, self.policies), f1)

    def load(self, path_to_dir):
        with open(os.path.join(path_to_dir, self.name + '_replay_buffer.pkl'), 'rb') as f1:
            self.observations, self.returns, self.policies = pickle.load(f1)
        self._update_normalization_stats()