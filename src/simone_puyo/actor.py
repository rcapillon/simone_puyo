import numpy as np
from multiprocessing import Pool

from .mcts import Node, run_mcts
from .utils import random_argmax_in_array
from .replay import ReplayBuffer, EpisodeBuffer
from .puyo import GAMEOVER_REWARD, get_chance_code


class Actor:
    """
    Actor class containing the neural network agent, the Puyo game and a replay buffer memory
    """
    def __init__(self, agent, game, agent_config, mcts_config, replay_config):
        self.agent = agent
        self.game = game
        self.agent_config = agent_config
        self.mcts_config = mcts_config
        self.replay_config = replay_config
        self.replay_buffer = ReplayBuffer(config=replay_config)

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
        self.replay_buffer.trim_buffer()

    def collect_game(self):
        """
        play a complete episode, storing collected samples in the replay buffer
        """
        episode_buffer = EpisodeBuffer(self.mcts_config.discount_factor)
        observation = self.reset_game()

        root = Node(
            reward=0.,
            done=False,
            agent=self.agent,
            game=self.game,
            parent=None,
            config=self.mcts_config
        )

        step = 1
        total_reward = 0.
        done = False
        while not done:
            legal_actions = self.game.get_legal_actions()
            _, policy, root = run_mcts(self.agent, self.game, config=self.mcts_config, root=root, training=True)
            random_index = random_argmax_in_array(policy[legal_actions])
            action = legal_actions[random_index]
            new_tsumo = [int(p) for p in self.game.state.queue.queue[2, :]]
            chance_code = get_chance_code(new_tsumo)

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

            episode_buffer.store_transition(observation, reward, policy)
            observation = next_observation
            root = new_root

            step += 1

        if done:
            if self.game.state.board.check_gameover():
                terminal_reward = GAMEOVER_REWARD
                episode_buffer.store_terminal_state(observation, terminal_reward)

        return episode_buffer, total_reward

    def collect_games_parallel(self, n_cpu=1):
        rewards = []
        pool = Pool(n_cpu)
        for episode_buffer, reward in pool.starmap(self.collect_game, [()] * n_cpu):
            self.replay_buffer.add_episode(episode_buffer)
            self.replay_buffer.trim_buffer()
            rewards.append(reward)

        return rewards

    def train_on_batch(self, epochs=1, verbose=2):
        """
        train agent for a single step from a random sample batch
        """
        batch_observations, batch_returns, batch_policies = self.replay_buffer.sample_batch(
            self.agent_config.batch_size
        )
        self.agent.train(batch_observations, batch_returns, batch_policies, epochs=epochs, verbose=verbose)

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

            observation, reward, done = self.game.step(action)
            total_reward += reward
            if reward > best_reward:
                best_reward = reward

            step += 1

        return best_reward, total_reward
