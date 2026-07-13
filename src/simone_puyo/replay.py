import os
import numpy as np
import pickle
from dataclasses import dataclass

from .puyo import can_mirror_observation, mirror_observation, mirror_policy


@dataclass
class ReplayConfig:
    max_capacity: int = 1000
    normalize_returns: bool = False
    value_target_mix: float = 0.5

    # --- Prioritized Experience Replay ---
    use_per: bool = True
    per_alpha: float = 0.6            # 0 = uniforme, 1 = priorite pure
    per_beta_start: float = 0.4       # correction importance-sampling au debut
    per_beta_end: float = 1.0         # ... en fin d'entrainement
    per_beta_anneal_steps: int = 200000   # nb d'appels a sample_batch() pour interpoler beta
    per_epsilon: float = 1e-3         # evite priorite nulle (transition jamais rechoisie)

    use_symmetry_augmentation: bool = True

    def __post_init__(self):
        assert 0.0 <= self.value_target_mix <= 1.0, "value_target_mix doit être dans [0, 1]"
        assert 0.0 <= self.per_alpha <= 1.0
        assert 0.0 <= self.per_beta_start <= self.per_beta_end <= 1.0


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
    def __init__(self, name, config=ReplayConfig()):
        self.name = name
        self.config = config

        self._observations = None
        self._returns = None
        self._policies = None
        self._priorities = None      # NOUVEAU

        self._write_idx = 0
        self._size = 0
        self._max_priority = 1.0     # NOUVEAU : priorite donnee aux nouvelles transitions
        self._sample_step = 0        # NOUVEAU : pour l'annealing de beta

    def __len__(self):
        return self._size

    def _allocate(self, observation_shape, n_actions):
        capacity = self.config.max_capacity
        self._observations = np.zeros((capacity,) + observation_shape, dtype=np.float32)
        self._returns = np.zeros(capacity, dtype=np.float32)
        self._policies = np.zeros((capacity, n_actions), dtype=np.float32)
        self._priorities = np.full(capacity, self._max_priority, dtype=np.float32)  # NOUVEAU

    @staticmethod
    def _augment_with_mirror(observations, targets, policies):
        """
        Double les transitions dont la case de game over (row=1, col=2) est
        libre, en ajoutant leur symetrique horizontal (observation + policy
        miroir). La cible de valeur est inchangee : le retour/valeur ne
        depend pas de l'orientation gauche-droite du plateau.
        """
        mirrorable = np.array(
            [can_mirror_observation(obs) for obs in observations], dtype=bool
        )

        if not np.any(mirrorable):
            return observations, targets, policies

        mirrored_observations = np.stack(
            [mirror_observation(obs) for obs in observations[mirrorable]]
        )
        mirrored_policies = np.stack(
            [mirror_policy(p) for p in policies[mirrorable]]
        )
        mirrored_targets = targets[mirrorable]

        all_observations = np.concatenate([observations, mirrored_observations], axis=0)
        all_targets = np.concatenate([targets, mirrored_targets], axis=0)
        all_policies = np.concatenate([policies, mirrored_policies], axis=0)

        return all_observations, all_targets, all_policies

    def add_episode(self, episode):
        episode.compute_returns()

        observations = np.asarray(episode.observations, dtype=np.float32)
        policies = np.asarray(episode.policies, dtype=np.float32)
        returns = np.asarray(episode.returns, dtype=np.float32)
        values = np.asarray(episode.values, dtype=np.float32)

        mix = self.config.value_target_mix
        blended_targets = (1 - mix) * returns + mix * values

        if self.config.use_symmetry_augmentation:
            observations, blended_targets, policies = self._augment_with_mirror(
                observations, blended_targets, policies
            )

        n_new = len(observations)
        if n_new == 0:
            return

        if self._observations is None:
            self._allocate(observations.shape[1:], policies.shape[1])

        capacity = self.config.max_capacity
        if n_new > capacity:
            observations = observations[-capacity:]
            policies = policies[-capacity:]
            blended_targets = blended_targets[-capacity:]
            n_new = capacity

        # Nouvelles transitions ecrites avec la priorite max courante,
        # pour garantir qu'elles soient echantillonnees au moins une fois
        # avant que leur vrai TD-error soit connu.
        new_priorities = np.full(n_new, self._max_priority, dtype=np.float32)

        end_idx = self._write_idx + n_new
        if end_idx <= capacity:
            self._observations[self._write_idx:end_idx] = observations
            self._returns[self._write_idx:end_idx] = blended_targets
            self._policies[self._write_idx:end_idx] = policies
            self._priorities[self._write_idx:end_idx] = new_priorities          # NOUVEAU
        else:
            first_part = capacity - self._write_idx
            second_part = n_new - first_part

            self._observations[self._write_idx:] = observations[:first_part]
            self._returns[self._write_idx:] = blended_targets[:first_part]
            self._policies[self._write_idx:] = policies[:first_part]
            self._priorities[self._write_idx:] = new_priorities[:first_part]    # NOUVEAU

            self._observations[:second_part] = observations[first_part:]
            self._returns[:second_part] = blended_targets[first_part:]
            self._policies[:second_part] = policies[first_part:]
            self._priorities[:second_part] = new_priorities[first_part:]       # NOUVEAU

        self._write_idx = end_idx % capacity
        self._size = min(self._size + n_new, capacity)

    def _current_beta(self):
        frac = min(1.0, self._sample_step / self.config.per_beta_anneal_steps)
        return self.config.per_beta_start + frac * (self.config.per_beta_end - self.config.per_beta_start)

    def sample_batch(self, batch_size):
        if self._size == 0:
            raise ValueError("Le replay buffer est vide.")

        batch_size = min(batch_size, self._size)

        if not self.config.use_per:
            indices = np.random.choice(self._size, size=batch_size, replace=False)
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            priorities = self._priorities[:self._size] ** self.config.per_alpha
            probs = priorities / priorities.sum()

            # Avec remise : standard pour PER, evite le cout d'un tirage sans
            # remise pondere (couteux) et permet a une transition tres prioritaire
            # d'apparaitre plusieurs fois dans le meme batch.
            indices = np.random.choice(self._size, size=batch_size, replace=True, p=probs)

            beta = self._current_beta()
            weights = (self._size * probs[indices]) ** (-beta)
            weights /= weights.max()  # normalisation, stabilite du gradient
            self._sample_step += 1

        batch_observations = self._observations[indices]
        batch_returns = self._returns[indices]
        batch_policies = self._policies[indices]

        if self.config.normalize_returns:
            valid_returns = self._returns[:self._size]
            mean = valid_returns.mean()
            std = valid_returns.std() + 1e-8
            batch_returns = (batch_returns - mean) / std

        return batch_observations, batch_returns, batch_policies, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        """
        A appeler apres chaque train_on_batch avec les TD-errors (return - value predite)
        des echantillons du batch, pour rafraichir leurs priorites.
        """
        priorities = np.abs(td_errors).astype(np.float32) + self.config.per_epsilon
        self._priorities[indices] = priorities
        self._max_priority = max(self._max_priority, float(priorities.max()))

    def save(self, path_to_dir):
        with open(os.path.join(path_to_dir, self.name + '_replay_buffer.pkl'), 'wb') as f1:
            pickle.dump(
                (self._observations, self._returns, self._policies, self._priorities,
                 self._write_idx, self._size, self._max_priority),
                f1
            )

    def load(self, path_to_dir):
        with open(os.path.join(path_to_dir, self.name + '_replay_buffer.pkl'), 'rb') as f1:
            data = pickle.load(f1)

        if len(data) == 7:
            (self._observations, self._returns, self._policies, self._priorities,
             self._write_idx, self._size, self._max_priority) = data
        else:
            # Migration depuis un ancien format sans priorites
            (self._observations, self._returns, self._policies,
             self._write_idx, self._size) = data
            capacity = self._observations.shape[0]
            self._max_priority = 1.0
            self._priorities = np.full(capacity, self._max_priority, dtype=np.float32)