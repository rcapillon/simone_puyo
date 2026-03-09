import os
import numpy as np
import pickle
from dataclasses import dataclass


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
        store_transition précédent. Valeur cible de cet état = 0.
        """
        self.observations.append(observation)
        self.rewards.append(0.)
        self.policies.append(np.zeros((22,)))

    def compute_returns(self, n_steps=None, bootstrap_values=None):
        """
        Calcule les retours cumulés pour tout l'épisode.

        Sans bootstrap (n_steps=None) : Monte Carlo pur jusqu'à la fin.

        Avec bootstrap :
            G_t^(n) = Σ_{k=0}^{n-1} γ^k · r_{t+k}  +  γ^n · V_θ(s_{t+n})

        Le bootstrap est supprimé proprement dans deux cas :
          - t+n dépasse la fin de l'épisode : on utilise les rewards
            restants sans bootstrap (pas de valeur disponible).
          - s_{t+n} est un état terminal (reward == 0 et dernier état
            après game over) : bootstrap_values[t+n] doit être 0,
            ce qui est garanti car store_terminal_state stocke reward=0
            et l'inférence réseau sur cet état converge vers 0.
        """
        T = len(self.rewards)
        self.returns = [0.] * T

        use_bootstrap = (n_steps is not None) and (bootstrap_values is not None)

        if not use_bootstrap:
            # Monte Carlo pur — comportement original
            value = 0.
            for i in reversed(range(T)):
                value = self.rewards[i] + self.discount_factor * value
                self.returns[i] = value
            return

        for t in range(T):
            g = 0.
            # Somme des rewards sur n steps (ou jusqu'à la fin)
            for k in range(n_steps):
                if t + k < T:
                    g += (self.discount_factor ** k) * self.rewards[t + k]
                else:
                    break

            # Bootstrap avec V_θ(s_{t+n}) si t+n est dans l'épisode
            bootstrap_idx = t + n_steps
            if bootstrap_idx < T:
                g += (self.discount_factor ** n_steps) * bootstrap_values[bootstrap_idx]
            # Sinon : pas de bootstrap, on a déjà accumulé tous les rewards
            # jusqu'à la fin — équivalent à V=0 au-delà de l'épisode.

            self.returns[t] = g


class ReplayBuffer:
    def __init__(self, name, config=ReplayConfig()):
        self.name = name
        self.config = config
        self.observations = []
        self.returns = []
        self.policies = []

        self._returns_mean = 0.
        self._returns_std = 1.

    def _update_normalization_stats(self):
        if len(self.returns) < 2:
            return
        arr = np.array(self.returns)
        self._returns_mean = float(arr.mean())
        self._returns_std = float(arr.std()) + 1e-8

    def add_episode(self, episode):
        """
        Ajoute un épisode au buffer.
        compute_returns() doit avoir été appelé AVANT add_episode,
        car le bootstrap nécessite les valeurs réseau calculées dans collect_game.
        """
        if not episode.returns:
            # Fallback sécurité : Monte Carlo pur si compute_returns
            # n'a pas été appelé explicitement
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

        batch_returns = (batch_returns - self._returns_mean) / self._returns_std

        return batch_observations, batch_returns, batch_policies

    def save(self, path_to_dir):
        with open(os.path.join(path_to_dir, self.name + '_replay_buffer.pkl'), 'wb') as f1:
            pickle.dump((self.observations, self.returns, self.policies), f1)

    def load(self, path_to_dir):
        with open(os.path.join(path_to_dir, self.name + '_replay_buffer.pkl'), 'rb') as f1:
            self.observations, self.returns, self.policies = pickle.load(f1)
        self._update_normalization_stats()