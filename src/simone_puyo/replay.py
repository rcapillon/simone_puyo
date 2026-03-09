import os
import numpy as np
import pickle
from dataclasses import dataclass


@dataclass
class ReplayConfig:
    """
    Dataclass for configuring the replay buffer.
    max_capacity : nombre maximum de transitions stockées.
    Les plus anciennes sont supprimées en premier (FIFO).
    Recommandé : 50 000 minimum pour un bon ratio diversité/mémoire.
    """
    max_capacity: int = 100000

    def __post_init__(self):
        pass


class EpisodeBuffer:
    """
    Stocke un épisode complet (une partie de Puyo).
    compute_returns() doit être appelé explicitement avant add_episode(),
    car le bootstrapping nécessite les valeurs réseau calculées dans collect_game.
    """
    def __init__(self, discount_factor):
        self.observations    = []
        self.rewards         = []
        self.returns         = []
        self.policies        = []
        self.discount_factor = discount_factor

    def store_transition(self, observation, reward, policy):
        """
        Stocke une transition (s_t, r_t, π_t).
        La policy doit être déjà masquée et renormalisée sur les actions légales
        (fait dans collect_game avant cet appel).
        """
        self.observations.append(observation)
        self.rewards.append(reward)
        self.policies.append(policy)

    def store_terminal_state(self, observation):
        """
        Stocke l'état terminal en cas de game over uniquement.

        reward = 0 : la pénalité GAMEOVER_REWARD est déjà dans le
        store_transition précédent (via game.step). Valeur cible = 0.

        Ne pas appeler en cas de fin par max_moves : la valeur résiduelle
        de ce dernier état n'est pas nulle et stocker reward=0 introduirait
        un biais négatif.
        """
        self.observations.append(observation)
        self.rewards.append(0.)
        self.policies.append(np.zeros((22,), dtype=np.float32))

    def compute_returns(self, n_steps=None, bootstrap_values=None):
        """
        Calcule les retours cumulés actualisés pour tout l'épisode.

        Sans bootstrap (n_steps=None) :
            Monte Carlo pur — retour calculé jusqu'à la fin de l'épisode.
            G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...

        Avec bootstrap (n_steps entier, bootstrap_values tableau numpy) :
            N-step return bootstrappé :
            G_t^(n) = Σ_{k=0}^{n-1} γ^k · r_{t+k}  +  γ^n · V_θ(s_{t+n})

            Deux cas de troncature propre :
            1. t+n dépasse la fin de l'épisode : on utilise les rewards
               restants sans bootstrap (équivalent V=0 au-delà).
            2. s_{t+n} est l'état terminal (game over) : bootstrap_values[t+n]
               converge vers 0 au fil de l'entraînement, ce qui est correct.

        Les bootstrap_values doivent être dans l'espace brut des rewards
        (dénormalisées si le réseau prédit dans un espace normalisé).
        """
        T            = len(self.rewards)
        self.returns = [0.] * T

        use_bootstrap = (n_steps is not None) and (bootstrap_values is not None)

        if not use_bootstrap:
            # Monte Carlo pur
            value = 0.
            for i in reversed(range(T)):
                value          = self.rewards[i] + self.discount_factor * value
                self.returns[i] = value
            return

        for t in range(T):
            g = 0.
            # Somme des rewards sur n steps (ou jusqu'à la fin de l'épisode)
            for k in range(n_steps):
                if t + k < T:
                    g += (self.discount_factor ** k) * self.rewards[t + k]
                else:
                    break

            # Bootstrap avec V_θ(s_{t+n}) si t+n est dans l'épisode
            bootstrap_idx = t + n_steps
            if bootstrap_idx < T:
                g += (self.discount_factor ** n_steps) * bootstrap_values[bootstrap_idx]
            # Sinon : pas de bootstrap — rewards accumulés jusqu'à la fin suffisent

            self.returns[t] = g


class ReplayBuffer:
    """
    Replay buffer circulaire (FIFO) avec normalisation des returns.

    Les statistiques de normalisation (mean, std) sont recalculées sur
    l'ensemble du buffer après chaque modification, ce qui est plus stable
    qu'une normalisation par batch (les stats évoluent lentement).
    """
    def __init__(self, name, config=ReplayConfig()):
        self.name   = name
        self.config = config

        self.observations = []
        self.returns      = []
        self.policies     = []

        # Statistiques de normalisation des returns.
        # Initialisées à neutre (mean=0, std=1) — opération neutre durant le warm-up.
        self._returns_mean = 0.
        self._returns_std  = 1.

    def _update_normalization_stats(self):
        """
        Recalcule mean et std sur l'ensemble des returns du buffer.
        Appelé après chaque modification (add_episode, trim_buffer, load).
        """
        if len(self.returns) < 2:
            return
        arr = np.array(self.returns)
        self._returns_mean = float(arr.mean())
        self._returns_std  = float(arr.std()) + 1e-8

    def add_episode(self, episode):
        """
        Ajoute un épisode au buffer.
        compute_returns() doit avoir été appelé sur l'épisode AVANT cet appel
        (fait dans collect_game). Si ce n'est pas le cas, fallback Monte Carlo pur.
        """
        if not episode.returns:
            # Fallback de sécurité
            episode.compute_returns()

        self.observations.extend(episode.observations)
        self.returns.extend(episode.returns)
        self.policies.extend(episode.policies)
        self._update_normalization_stats()

    def trim_buffer(self):
        """
        Supprime les transitions les plus anciennes si le buffer dépasse max_capacity.
        """
        current_size = len(self.observations)
        if current_size > self.config.max_capacity:
            n_removed         = current_size - self.config.max_capacity
            self.observations = self.observations[n_removed:]
            self.returns      = self.returns[n_removed:]
            self.policies     = self.policies[n_removed:]
            self._update_normalization_stats()

    def sample_batch(self, batch_size):
        """
        Tire aléatoirement un batch de transitions sans remise.
        Les returns sont normalisés avec les stats du buffer complet :
        - Stabilise les gradients du value head
        - L'UCT compare les Q-values entre elles → le changement d'échelle
          ne dégrade pas la sélection d'action dans le MCTS
        """
        current_size = len(self.observations)
        if current_size < batch_size:
            batch_size = current_size

        indices = np.random.choice(current_size, size=batch_size, replace=False)

        batch_observations = np.array(self.observations)[indices, :, :, :]
        batch_returns      = np.array(self.returns)[indices]
        batch_policies     = np.array(self.policies)[indices, :]

        # Normalisation sur les stats du buffer complet
        batch_returns = (batch_returns - self._returns_mean) / self._returns_std

        return batch_observations, batch_returns, batch_policies

    def save(self, path_to_dir):
        """
        Sauvegarde le contenu du buffer dans un fichier pickle.
        """
        os.makedirs(path_to_dir, exist_ok=True)
        with open(os.path.join(path_to_dir, self.name + '_replay_buffer.pkl'), 'wb') as f:
            pickle.dump((self.observations, self.returns, self.policies), f)

    def load(self, path_to_dir):
        """
        Charge le contenu du buffer depuis un fichier pickle.
        Recalcule les stats de normalisation après chargement.
        """
        with open(os.path.join(path_to_dir, self.name + '_replay_buffer.pkl'), 'rb') as f:
            self.observations, self.returns, self.policies = pickle.load(f)
        self._update_normalization_stats()