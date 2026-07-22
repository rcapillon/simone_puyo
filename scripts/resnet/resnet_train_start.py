import os
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import pickle

from src.simone_puyo.agent import ResNetConfig, ResNetAgent
from src.simone_puyo.puyo import PuyoGame
from src.simone_puyo.mcts import MCTSConfig
from src.simone_puyo.replay import ReplayConfig
from src.simone_puyo.actor import Actor, RewardConfig


if __name__ == '__main__':
    agent_name = 'resnet'
    agent_config = ResNetConfig(
        num_res_blocks=10,
        num_filters=128,
        kernel_size=3,
        policy_filters=3,
        policy_hidden_size=256,
        value_filters=2,
        value_hidden_size=256,
        l2_regularization=1.5e-4,
        use_batch_norm=True,
        learning_rate=3e-4,
        batch_size=256,
        value_loss_weight=1.,
        policy_loss_weight=1.
    )
    agent = ResNetAgent(name=agent_name, config=agent_config)
    agent.build_model(summary=True)

    max_moves = 20
    puyo_game = PuyoGame(max_moves=max_moves)

    mcts_config = MCTSConfig(
        n_simulations=2000,
        UCT_exploration_constant=2.,
        discount_factor=0.995,
        dirichlet_alpha=0.5,
        dirichlet_epsilon=0.25,
        tau_max=2.5,
        tau_min=0.3,
        batch_size=32,
        virtual_loss=2.
    )

    replay_config = ReplayConfig(
        max_capacity=50000
    )

    reward_config = RewardConfig(
        use_potential_shaping=True,
        potential_shaping_weight=1.
    )

    actor = Actor(agent, puyo_game, agent_config, mcts_config, replay_config, reward_config)

    # TRAINING / TEST CYCLES
    n_workers = 4
    n_cycles = 1
    training_cycles = 10
    collect_cycles = 1
    gradient_steps_per_cycle = 20
    buffer_min_length = 5000

    collected_rewards = []

    t0 = time()
    for i in range(n_cycles):
        print(f'CYCLE {i + 1}')
        batch_counter = 1
        # SAMPLE COLLECTION AND TRAINING LOOP
        for _ in range(training_cycles):
            for _ in range(collect_cycles):
                t_episode_0 = time()
                print(f'EPISODE BATCH {batch_counter}')
                rewards = actor.collect_games_parallel(n_workers=n_workers)
                collected_rewards.extend(rewards)
                batch_counter += 1
                t_episode_1 = time()
                print(f'Episode batch took: {t_episode_1 - t_episode_0} seconds.')
            if (actor.replay_buffer.__len__() >= agent_config.batch_size
                    and actor.replay_buffer.__len__() >= buffer_min_length):
                for _ in range(gradient_steps_per_cycle):
                    actor.train_on_batch()

        # TEST
        print('TEST GAMES')
        n_test_games = 100
        test_best_rewards = []
        test_total_rewards = []
        for _ in tqdm(range(n_test_games)):
            best_reward, total_reward = actor.play_test_game()
            test_best_rewards.append(best_reward)
            test_total_rewards.append(total_reward)
        average_test_best_reward = np.mean(test_best_rewards)
        average_test_total_reward = np.mean(test_total_rewards)
        actor.agent.test_scores.append((average_test_best_reward, average_test_total_reward))
    t1 = time()

    print(f'Time elapsed: {t1 - t0} seconds.')

    # training loss plot
    _, ax = plt.subplots()
    value_head_loss = [losses[0] for losses in actor.agent.training_loss]
    ax.semilogy(value_head_loss, label='value head loss')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Loss')
    ax.set_title('Value head loss')
    plt.savefig('./' + agent_name + '_value_loss.png')

    _, ax = plt.subplots()
    policy_head_loss = [losses[1] for losses in actor.agent.training_loss]
    ax.semilogy(policy_head_loss, label='policy head loss')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Loss')
    ax.set_title('Policy head loss')
    plt.savefig('./' + agent_name + '_policy_loss.png')

    # test plot
    _, ax = plt.subplots()
    test_best_rewards = [test_scores[0] for test_scores in actor.agent.test_scores]
    test_total_rewards = [test_scores[1] for test_scores in actor.agent.test_scores]
    ax.plot(test_best_rewards, label='best reward')
    ax.plot(test_total_rewards, label='total reward')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Test cycles')
    ax.set_ylabel('Average reward')
    ax.set_title('Average test rewards (no MCTS)')
    plt.savefig('./' + agent_name + '_test_rewards.png')

    _, ax = plt.subplots()
    test_best_rewards = [test_scores[0] for test_scores in actor.agent.test_scores]
    ax.plot(test_best_rewards)
    ax.grid()
    ax.legend()
    ax.set_xlabel('Test cycles')
    ax.set_ylabel('Average reward')
    ax.set_title('Average best reward per game (no MCTS)')
    plt.savefig('./' + agent_name + '_test_best_rewards.png')

    _, ax = plt.subplots()
    ax.plot(collected_rewards)
    ax.grid()
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Collected rewards')
    ax.set_title('Rewards collected during training')
    plt.savefig('./' + agent_name + '_collected_rewards.png')

    # save
    replay_path_to_dir = '../../saved_data/'
    agent_path_to_dir = '../../saved_agents/'
    actor.agent.save_model(agent_path_to_dir)
    actor.replay_buffer.save(replay_path_to_dir)
    with open(os.path.join('../../saved_data/', agent_name + '_collected_rewards.pkl'), 'wb') as f:
        pickle.dump(collected_rewards, f)
