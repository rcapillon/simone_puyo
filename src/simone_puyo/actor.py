import numpy as np
from multiprocessing import Pool

from .mcts_batched import Node, run_mcts
from .utils import random_argmax_in_array
from .replay import ReplayBuffer, EpisodeBuffer
from .puyo import get_chance_code, reward_dict


class Actor:
    """
    Actor class containing the neural network agent, the Puyo game and a replay buffer memory.
    """
    def __init__(self, agent, game, agent_config, mcts_config, replay_config):
        self.agent        = agent
        self.game         = game
        self.agent_config = agent_config
        self.mcts_config  = mcts_config
        self.replay_config = replay_config
        self.replay_buffer = ReplayBuffer(name=self.agent.name, config=replay_config)

    def reset_game(self):
        """
        Resets the Puyo game.
        """
        return self.game.reset()

    def load_replay_buffer(self, path_to_dir):
        """
        Load replay buffer memory from pickle file.
        """
        self.replay_buffer.load(path_to_dir)
        self.replay_buffer.trim_buffer()

    def collect_game(self):
        """
        Play a complete episode, storing collected samples in an EpisodeBuffer.
        Computes n-step bootstrapped returns using a single batched inference
        over all episode observations at the end of the episode.
        """
        episode_buffer = EpisodeBuffer(self.mcts_config.discount_factor)
        observation    = self.reset_game()

        root = Node(
            reward=0.,
            done=False,
            agent=self.agent,
            game=self.game,
            parent=None,
            config=self.mcts_config
        )

        step         = 1
        total_reward = 0.
        done         = False

        while not done:
            legal_actions = self.game.get_legal_actions()
            _, policy, root = run_mcts(
                self.agent, self.game,
                config=self.mcts_config,
                root=root,
                training=True
            )

            random_index = random_argmax_in_array(policy[legal_actions])
            action       = legal_actions[random_index]
            new_tsumo    = [int(p) for p in self.game.state.queue.queue[2, :]]
            chance_code  = get_chance_code(new_tsumo)

            next_observation, reward, done = self.game.step(action)
            total_reward += reward

            try:
                new_root = root.children[(action, chance_code)]
                new_root.parent = None
            except KeyError:
                new_root = Node(
                    reward=reward,
                    done=done,
                    agent=self.agent,
                    game=self.game,
                    parent=None,
                    config=self.mcts_config
                )

            # Masquage et renormalisation de la policy MCTS sur les actions légales.
            # Evite que la cross-entropy pénalise les actions légales via le signal
            # contradictoire des actions illégales dans la normalisation softmax.
            policy_stored          = np.zeros(22, dtype=np.float32)
            policy_stored[legal_actions] = policy[legal_actions]
            total_p = policy_stored.sum()
            if total_p > 0:
                policy_stored /= total_p

            episode_buffer.store_transition(observation, reward, policy_stored)
            observation = next_observation
            root        = new_root
            step       += 1

        # Stocke l'état terminal uniquement en cas de game over.
        # - next_observation est le board avec la cellule gameover remplie → valeur vraie = 0
        # - GAMEOVER_REWARD est déjà dans le store_transition précédent
        # Ne pas stocker en cas de fin par max_moves : la valeur résiduelle n'est pas nulle.
        if self.game.state.board.check_gameover():
            episode_buffer.store_terminal_state(observation)

        # ------------------------------------------------------------------
        # Inférence batchée pour les valeurs bootstrap.
        #
        # Utilise node._cached_input via episode_buffer.observations :
        # les observations ont été copiées depuis next_observation (= game.get_input())
        # à chaque step — même données que _cached_input sur les nodes correspondants.
        #
        # Une seule passe réseau sur toutes les T observations de l'épisode,
        # bien plus efficace que T appels individuels pendant la boucle.
        # ------------------------------------------------------------------
        obs_array        = np.array(episode_buffer.observations)   # (T, 14, 6, 4)
        bootstrap_values, _ = self.agent(obs_array)                # (T,) normalisé
        bootstrap_values = np.array(bootstrap_values).flatten()

        # Dénormalisation : le réseau prédit dans l'espace normalisé du buffer
        # (mean≈0, std≈1), mais les rewards stockés sont bruts.
        # Repasser dans l'espace brut avant compute_returns(),
        # puis sample_batch() renormalisera les returns finaux.
        # Durant le warm-up (_returns_std=1, _returns_mean=0), opération neutre.
        bootstrap_values = (
            bootstrap_values * self.replay_buffer._returns_std
            + self.replay_buffer._returns_mean
        )

        episode_buffer.compute_returns(
            n_steps=self.mcts_config.n_steps,
            bootstrap_values=bootstrap_values
        )

        return episode_buffer, total_reward

    def collect_games_parallel(self, n_cpu=1):
        """
        Collecte n_cpu épisodes en parallèle.
        Utilise un context manager pour garantir la fermeture du pool
        même en cas d'exception.

        Note : fonctionne sur Linux (fork par défaut — le modèle Keras est
        accessible dans les workers sans sérialisation). Sur macOS (Python 3.8+)
        ou Windows (spawn par défaut), un refactoring avec envoi explicite des
        poids serait nécessaire.
        """
        with Pool(processes=n_cpu) as pool:
            results = pool.starmap(self.collect_game, [()] * n_cpu)

        rewards = []
        for episode_buffer, reward in results:
            self.replay_buffer.add_episode(episode_buffer)
            self.replay_buffer.trim_buffer()
            rewards.append(reward)

        return rewards

    def train_on_batch(self, epochs=1, verbose=0):
        """
        Train agent for a single gradient step from a freshly sampled batch.
        Toujours appeler avec une nouvelle batch (ne jamais réutiliser la même).
        """
        batch_observations, batch_returns, batch_policies = self.replay_buffer.sample_batch(
            self.agent_config.batch_size
        )
        self.agent.train(batch_observations, batch_returns, batch_policies, epochs=epochs, verbose=verbose)

    def play_test_game(self):
        """
        Play a complete episode without MCTS, greedy deterministic policy.
        Used for evaluating the agent between training cycles.

        Returns:
            best_chain   (int)   : longest chain achieved during the game
            total_reward (float) : sum of all rewards over the episode
        """
        best_chain   = 0
        total_reward = 0.

        observation = self.reset_game()
        done        = False

        while not done:
            legal_actions = self.game.get_legal_actions()
            _, policy     = self.agent(observation)

            # Sélection déterministe sur les actions légales uniquement.
            # argmax pur (pas de random tie-breaking) pour une évaluation
            # reproductible et représentative de la politique apprise.
            masked_policy = np.full(22, -np.inf, dtype=np.float32)
            masked_policy[legal_actions] = policy[legal_actions]
            action = int(np.argmax(masked_policy))

            observation, reward, done = self.game.step(action)
            total_reward += reward

            # Retrouver la chain length depuis le reward
            # reward_dict[k] = sqrt(k^2.5 + 1) - 1
            if reward > 0:
                for k in range(1, 20):
                    if abs(reward_dict[k] - reward) < 1e-6:
                        best_chain = max(best_chain, k)
                        break

        return best_chain, total_reward