import os
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import pickle

# enable_mixed_precision() DOIT être appelé avant tout import Keras/TF
# et avant tout build_model(). Décommenter si GPU Turing/Ampere (RTX 20xx+).
# from src.simone_puyo.agents import enable_mixed_precision
# enable_mixed_precision()

from src.simone_puyo.agents import ResNetConfig, ResNetAgent, MLPConfig, MLPAgent
from src.simone_puyo.puyo import PuyoGame
from src.simone_puyo.mcts_batched import MCTSConfig
from src.simone_puyo.replay import ReplayConfig
from src.simone_puyo.actor import Actor


# ======================================================================
# Curriculum n_steps
# ======================================================================

def get_n_steps(training_step, total_steps):
    """
    Curriculum sur n_steps pour le bootstrapping des returns.

    - Début (30% des steps) : quasi Monte Carlo pur (n_steps=18).
      Le value head est mal calibré → on l'expose à des returns MC fiables
      pour l'amorcer correctement.
    - Milieu (30-70%) : transition progressive (n_steps=15).
    - Fin (70%+) : bootstrapping plus agressif (n_steps=10).
      Le value head est plus précis → on profite de sa prédiction pour
      réduire la variance et accélérer la convergence.

    Pour des épisodes de 20 coups max, n_steps=18 est quasi-MC pur.
    """
    progress = training_step / max(total_steps, 1)
    if progress < 0.30:
        return 18
    elif progress < 0.70:
        return 15
    else:
        return 10


# ======================================================================
# Configuration
# ======================================================================

if __name__ == '__main__':
    agent_type = 'resnet'

    if agent_type == 'mlp':
        agent_name   = 'mlp_agent_1'
        agent_config = MLPConfig(
            n_common_hidden_layers=2,
            n_common_neurons_per_layer=512,
            n_value_hidden_layers=1,
            n_value_neurons_per_layer=512,
            n_policy_hidden_layers=1,
            n_policy_neurons_per_layer=512,
            learning_rate=1e-3,
            batch_size=256
        )
        agent = MLPAgent(name=agent_name, config=agent_config)

    elif agent_type == 'resnet':
        agent_name   = 'resnet_agent_1'
        agent_config = ResNetConfig(
            num_res_blocks=8,
            num_filters=128,
            kernel_size=3,
            policy_filters=4,
            policy_hidden_size=256,
            value_filters=2,
            value_hidden_size=256,
            l2_regularization=1e-4,
            use_batch_norm=True,
            learning_rate=1e-3,
            batch_size=256
        )
        agent = ResNetAgent(name=agent_name, config=agent_config)

    else:
        raise ValueError(f'Unknown agent type: {agent_type}')

    agent.build_model(summary=True)

    max_moves  = 20
    puyo_game  = PuyoGame(max_moves=max_moves)

    mcts_config = MCTSConfig(
        n_simulations=1000,
        batch_size=128,              # 1000/128 ≈ 8 rounds — bon équilibre GPU/qualité
        UCT_exploration_constant=1.5,
        discount_factor=0.99,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        tau_max=2.5,
        tau_min=0.3,
        virtual_loss=1.,
        n_steps=15                   # sera mis à jour dynamiquement par le curriculum
    )

    replay_config = ReplayConfig(
        max_capacity=50000
    )

    actor = Actor(agent, puyo_game, agent_config, mcts_config, replay_config)

    # ======================================================================
    # Hyperparamètres d'entraînement
    # ======================================================================
    n_cpu                  = 4
    n_cycles               = 1       # cycles principaux collect → train → test
    training_steps_per_cycle = 10     # itérations collect+train par cycle
    gradient_steps         = 4       # gradient steps par itération (batch indépendant à chaque fois)
    buffer_min_length      = 5000     # transitions minimales avant de commencer à entraîner
    n_test_games           = 100       # parties de test par cycle
    save_every_n_cycles    = 5        # sauvegarde tous les N cycles

    replay_path_to_dir = '../saved_data/'
    agent_path_to_dir  = '../saved_agents/'
    os.makedirs(replay_path_to_dir, exist_ok=True)
    os.makedirs(agent_path_to_dir, exist_ok=True)

    collected_rewards     = []

    # ======================================================================
    # Phase de warm-up : remplir le buffer avant tout entraînement
    # ======================================================================
    print('=' * 60)
    print('WARM-UP PHASE')
    print(f'Target buffer size: {buffer_min_length} transitions')
    print('=' * 60)

    warmup_batch = 0
    while len(actor.replay_buffer.observations) < buffer_min_length:
        rewards = actor.collect_games_parallel(n_cpu=n_cpu)
        collected_rewards.extend(rewards)
        warmup_batch += 1
        current_size = len(actor.replay_buffer.observations)
        print(f'  Warm-up batch {warmup_batch}: buffer {current_size}/{buffer_min_length}')

    print(f'Warm-up complete — {len(actor.replay_buffer.observations)} transitions in buffer.\n')

    # ======================================================================
    # Boucle principale collect → train → test
    # ======================================================================
    t0 = time()

    for cycle in range(n_cycles):
        print('=' * 60)
        print(f'CYCLE {cycle + 1}/{n_cycles}')
        print('=' * 60)

        # ----------------------------------------------------------------
        # Collect + Train
        # ----------------------------------------------------------------
        for step in range(training_steps_per_cycle):

            # Collecte
            t_col = time()
            rewards = actor.collect_games_parallel(n_cpu=n_cpu)
            collected_rewards.extend(rewards)
            t_col = time() - t_col

            # Gradient steps — batch rééchantillonné à chaque fois
            t_train = time()
            for _ in range(gradient_steps):
                actor.train_on_batch(epochs=1, verbose=0)
            t_train = time() - t_train

            print(
                f'  Step {step + 1}/{training_steps_per_cycle} | '
                f'n_steps={actor.mcts_config.n_steps} | '
                f'buf={len(actor.replay_buffer.observations)} | '
                f'collect={t_col:.1f}s | train={t_train:.1f}s'
            )

        # ----------------------------------------------------------------
        # Test sans MCTS (greedy déterministe)
        # ----------------------------------------------------------------
        print(f'\n  TEST ({n_test_games} games, no MCTS)')
        test_best_chains   = []
        test_total_rewards = []

        for _ in tqdm(range(n_test_games), ncols=60):
            best_chain, total_reward = actor.play_test_game()
            test_best_chains.append(best_chain)
            test_total_rewards.append(total_reward)

        avg_best_chain   = float(np.mean(test_best_chains))
        avg_total_reward = float(np.mean(test_total_rewards))

        # Distribution des meilleures chaînes
        chain_counts = {}
        for c in test_best_chains:
            chain_counts[c] = chain_counts.get(c, 0) + 1
        dist_str = '  '.join(
            f'chain{k}:{v}' for k, v in sorted(chain_counts.items())
        )

        print(f'  Avg best chain  : {avg_best_chain:.2f}')
        print(f'  Avg total reward: {avg_total_reward:.2f}')
        print(f'  Distribution    : {dist_str}')

        actor.agent.test_scores.append((avg_best_chain, avg_total_reward))

        # ----------------------------------------------------------------
        # Sauvegarde périodique
        # ----------------------------------------------------------------
        if (cycle + 1) % save_every_n_cycles == 0:
            print(f'\n  Saving checkpoint at cycle {cycle + 1}...')
            actor.agent.save_model(agent_path_to_dir)
            actor.replay_buffer.save(replay_path_to_dir)
            with open(os.path.join(replay_path_to_dir, agent_name + '_collected_rewards.pkl'), 'wb') as f:
                pickle.dump(collected_rewards, f)
            print('  Checkpoint saved.')

        print()

    t1 = time()
    print(f'\nTotal training time: {(t1 - t0) / 60:.1f} minutes')

    # ======================================================================
    # Sauvegarde finale
    # ======================================================================
    print('Saving final model...')
    actor.agent.save_model(agent_path_to_dir)
    actor.replay_buffer.save(replay_path_to_dir)
    with open(os.path.join(replay_path_to_dir, agent_name + '_collected_rewards.pkl'), 'wb') as f:
        pickle.dump(collected_rewards, f)

    # ======================================================================
    # Plots
    # ======================================================================

    # Value head loss
    fig, ax = plt.subplots()
    value_losses = [losses[0] for losses in actor.agent.training_loss]
    ax.semilogy(value_losses, label='value head loss')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Gradient steps')
    ax.set_ylabel('Loss')
    ax.set_title('Value head loss')
    plt.savefig('./' + agent_name + '_value_loss.png')
    plt.close()

    # Policy head loss
    fig, ax = plt.subplots()
    policy_losses = [losses[1] for losses in actor.agent.training_loss]
    ax.semilogy(policy_losses, label='policy head loss')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Gradient steps')
    ax.set_ylabel('Loss')
    ax.set_title('Policy head loss')
    plt.savefig('./' + agent_name + '_policy_loss.png')
    plt.close()

    # Test scores : avg best chain + avg total reward
    fig, ax = plt.subplots()
    avg_best_chains   = [s[0] for s in actor.agent.test_scores]
    avg_total_rewards = [s[1] for s in actor.agent.test_scores]
    ax.plot(avg_best_chains,   label='avg best chain (no MCTS)')
    ax.plot(avg_total_rewards, label='avg total reward (no MCTS)')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Test cycles')
    ax.set_ylabel('Value')
    ax.set_title('Test scores (greedy, no MCTS)')
    plt.savefig('./' + agent_name + '_test_scores.png')
    plt.close()

    # Collected rewards during training
    fig, ax = plt.subplots()
    ax.plot(collected_rewards)
    ax.grid()
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Total reward')
    ax.set_title('Rewards collected during training (with MCTS)')
    plt.savefig('./' + agent_name + '_collected_rewards.png')
    plt.close()

    print('All plots saved.')