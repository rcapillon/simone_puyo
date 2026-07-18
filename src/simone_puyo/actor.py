import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .mcts import Node, run_mcts
from .utils import random_argmax_in_array
from .replay import ReplayBuffer, EpisodeBuffer
from .puyo import get_chance_code, reward_dict, chain_potential


@dataclass
class RewardConfig:
    """
    Reward shaping optionnel (potential-based, cf. Ng, Harada & Russell 1999) :
    ajoute gamma*Phi(s') - Phi(s) a la vraie recompense de chaine, avec Phi le
    potentiel de chaine latent (voir chain_potential dans puyo.py).
    N'affecte QUE la recompense stockee pour l'entrainement (episode_buffer) :
    ni MCTS (Node est toujours construit avec la vraie recompense), ni
    total_reward (score reel rapporte/trace), ni play_test_game(). Desactive
    par defaut : use_potential_shaping=False restaure exactement le systeme
    actuel (recompense = uniquement les chaines effectuees).
    """
    use_potential_shaping: bool = False
    potential_shaping_weight: float = 1.0


class Actor:
    """
    Actor class containing the neural network agent, the Puyo game and a replay buffer memory
    """
    def __init__(self, agent, game, agent_config, mcts_config, replay_config, reward_config=None):
        self.agent = agent
        self.game = game
        self.agent_config = agent_config
        self.mcts_config = mcts_config
        self.replay_config = replay_config
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        self.replay_buffer = ReplayBuffer(name=self.agent.name, config=replay_config)

    def reset_game(self):
        """
        resets the Puyo game
        """
        return self.game.reset()

    def load_replay_buffer(self, path_to_dir):
        """
        load replay buffer memory from pickle file
        """
        self.replay_buffer.load(path_to_dir)

    def collect_game(self, game=None):
        """
        play a complete episode, storing collected samples in the replay buffer.
        Si `game` est fourni, joue sur cette instance independante (utile pour
        le self-play parallele) ; sinon, utilise self.game comme avant.
        """
        if game is None:
            game = self.game

        episode_buffer = EpisodeBuffer(self.mcts_config.discount_factor)
        observation = game.reset()

        root = Node(
            reward=0.,
            done=False,
            terminal=False,
            agent=self.agent,
            game=game,
            parent=None,
            config=self.mcts_config
        )

        step = 1
        total_reward = 0.
        done = False
        while not done:
            legal_actions = root.legal_actions
            value, policy, root = run_mcts(self.agent, game, config=self.mcts_config, root=root, training=True)
            random_index = random_argmax_in_array(policy[legal_actions])
            action = legal_actions[random_index]

            if self.reward_config.use_potential_shaping:
                phi_s = reward_dict[min(chain_potential(game.state.board.num_board), 19)]

            next_observation, reward, done, gameover = game.step(action)
            new_tsumo = [int(p) for p in game.state.queue.queue[2, :]]
            chance_code = get_chance_code(new_tsumo)
            total_reward += reward  # total_reward reste la vraie recompense, jamais "shapee"

            stored_reward = reward
            if self.reward_config.use_potential_shaping:
                # Phi(s') = 0 sur un vrai game over (pas de potentiel futur),
                # calcule normalement en cas de troncature par max_moves --
                # meme logique que le bootstrap de troncature ci-dessous.
                phi_s_prime = 0. if gameover else reward_dict[min(chain_potential(game.state.board.num_board), 19)]
                shaping = self.reward_config.potential_shaping_weight * (
                    self.mcts_config.discount_factor * phi_s_prime - phi_s
                )
                stored_reward = reward + shaping

            try:
                new_root = root.children[(action, chance_code)]
                new_root.parent = None
            except KeyError:
                new_root = Node(
                    reward=reward,   # la vraie recompense : la recherche MCTS n'est jamais "shapee"
                    done=done,
                    terminal=gameover,
                    agent=self.agent,
                    game=game,
                    parent=None,
                    config=self.mcts_config
                )

            policy_legal = np.zeros(22)
            policy_legal[legal_actions] = policy[legal_actions]
            policy_legal /= policy_legal.sum()

            episode_buffer.store_transition(observation, stored_reward, policy_legal, value)
            observation = next_observation
            root = new_root

            step += 1

        if done and not gameover:
            # Partie tronquee par max_moves, pas un vrai game over : on
            # bootstrap avec l'estimation brute du reseau sur l'etat final
            # plutot que de supposer (a tort) un retour futur nul.
            bootstrap_value, _ = self.agent(next_observation)
            episode_buffer.bootstrap_value = float(bootstrap_value)

        return episode_buffer, total_reward

    def collect_games_parallel(self, n_workers=1):
        """
        Joue n_workers parties en parallele via des threads (pas des process) :
        le modele/agent est partage en memoire sans etre serialise -- ce qui
        evite le probleme de pickling de predict_fn -- et chaque partie recoit
        sa propre instance de jeu independante pour eviter toute concurrence
        d'ecriture sur self.game.
        """

        def _play_one_game():
            local_game = type(self.game)(self.game.max_moves)
            return self.collect_game(game=local_game)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(lambda _: _play_one_game(), range(n_workers)))

        rewards = []
        for episode_buffer, reward in results:
            self.replay_buffer.add_episode(episode_buffer)
            rewards.append(reward)

        return rewards

    def train_on_batch(self, epochs=1, verbose=2):
        """
        train agent for a single step from a prioritized sample batch
        """
        batch_observations, batch_returns, batch_policies, indices, weights = \
            self.replay_buffer.sample_batch(self.agent.config.batch_size)

        # TD-error = retour cible - valeur actuelle predite par le reseau,
        # utilise pour rafraichir les priorites (avant l'update des poids,
        # donc coherent avec l'erreur qui a motive ce tirage).
        predicted_values, _ = self.agent(batch_observations)
        predicted_values = np.asarray(predicted_values).reshape(-1)
        td_errors = batch_returns - predicted_values

        self.agent.train(
            batch_observations, batch_returns, batch_policies,
            sample_weight=weights, epochs=epochs, verbose=verbose
        )

        self.replay_buffer.update_priorities(indices, td_errors)

    def play_test_game(self):
        """
        play a complete episode without MCTS, used for evaluating the agent
        """
        best_reward = -np.inf
        total_reward = 0.

        observation = self.reset_game()

        step = 1
        done = False
        while not done:
            legal_actions = self.game.get_legal_actions()
            _, policy = self.agent(observation)
            random_index = random_argmax_in_array(policy[legal_actions])
            action = legal_actions[random_index]

            observation, reward, done, _ = self.game.step(action)
            total_reward += reward
            if reward > best_reward:
                best_reward = reward

            step += 1

        return best_reward, total_reward