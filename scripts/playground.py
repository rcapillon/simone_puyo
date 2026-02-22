import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt

from src.simone_puyo.agents import ResNetConfig, ResNetAgent
from src.simone_puyo.puyo import PuyoGame
from src.simone_puyo.mcts import MCTSConfig
from src.simone_puyo.replay import ReplayConfig
from src.simone_puyo.actor import Actor


if __name__ == '__main__':
    resnet_config = ResNetConfig(
        num_res_blocks=10,
        num_filters=256,
        kernel_size=3,
        policy_filters=2,
        policy_hidden_size=512,
        value_filters=1,
        value_hidden_size=512,
        l2_regularization=1e-4,
        use_batch_norm=True,
        learning_rate=1e-3,
        batch_size=64
    )

    resnet_agent = ResNetAgent(name='resnet_agent_1', config=resnet_config)
    resnet_agent.build_model(summary=True)

    max_moves = 20
    puyo_game = PuyoGame(max_moves=max_moves)

    mcts_config = MCTSConfig(
        n_simulations=200,
        UCT_exploration_constant=1.5,
        discount_factor=0.99,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        base_temperature=1.
    )

    replay_config = ReplayConfig(
        max_capacity=100000
    )

    actor = Actor(resnet_agent, puyo_game, resnet_config, mcts_config, replay_config)

    # TRAINING / TEST CYCLES
    n_cpu = 4
    n_cycles = 10
    episode_batches = 10
    buffer_min_length = 1000

    t0 = time()
    for i in range(n_cycles):
        print(f'CYCLE {i + 1}')

        # SAMPLE COLLECTION AND TRAINING LOOP
        for j in range(episode_batches):
            t_episode_0 = time()
            print(f'EPISODE BATCH {j + 1}')
            rewards = actor.collect_games_parallel(n_cpu=n_cpu)
            print(f'Average reward: {np.mean(rewards)}')
            if (len(actor.replay_buffer.observations) >= resnet_config.batch_size
                    and len(actor.replay_buffer.observations) >= buffer_min_length):
                actor.train_on_batch()
            t_episode_1 = time()
            print(f'Episode batch took: {t_episode_1 - t_episode_0} seconds.')

        # TEST
        print('TEST GAMES')
        n_test_games = 100
        test_rewards = []
        for _ in tqdm(range(n_test_games)):
            best_reward = actor.play_test_game()
            test_rewards.append(best_reward)
        average_test_reward = np.mean(test_rewards)
        actor.agent.test_scores.append(average_test_reward)
    t1 = time()

    print(f'Time elapsed: {t1 - t0} seconds.')

    # training loss plot
    _, ax = plt.subplots()
    value_head_loss = [losses[0] for losses in actor.agent.training_loss]
    policy_head_loss = [losses[1] for losses in actor.agent.training_loss]
    ax.semilogy(value_head_loss, label='value head loss')
    ax.semilogy(policy_head_loss, label='policy head loss')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training losses')
    plt.savefig('./testagent_training_losses.png')

    # test plot
    _, ax = plt.subplots()
    ax.plot(actor.agent.test_scores)
    ax.grid()
    ax.set_xlabel('Test cycles')
    ax.set_ylabel('Average reward')
    ax.set_title('Average test rewards (no MCTS)')
    plt.savefig('./testagent_test_rewards.png')

    # save
    replay_path_to_dir = '../saved_data/'
    agent_path_to_dir = '../saved_agents/'
    actor.agent.save_model(agent_path_to_dir)
    actor.replay_buffer.save(replay_path_to_dir)
