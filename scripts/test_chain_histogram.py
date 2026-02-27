import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.simone_puyo.agents import ResNetConfig, ResNetAgent, MLPConfig, MLPAgent
from src.simone_puyo.puyo import PuyoGame, GAMEOVER_REWARD, get_chance_code
from src.simone_puyo.mcts_batched import MCTSConfig, Node, run_mcts
from src.simone_puyo.replay import ReplayConfig
from src.simone_puyo.actor import Actor
from src.simone_puyo.utils import random_argmax_in_array


if __name__ == '__main__':
    agent_type = 'resnet'
    with_mcts = True

    if agent_type == 'resnet':
        agent = ResNetAgent(name=agent_type)
        agent_config = ResNetConfig()
    elif agent_type == 'mlp':
        agent = MLPAgent(name=agent_type)
        agent_config = MLPConfig()
    else:
        raise ValueError(f'Unknown agent type: {agent_type}')

    agent.load_model('../saved_agents', summary=False)

    max_moves = 20
    puyo_game = PuyoGame(max_moves=max_moves)

    n_simulations = 1000
    mcts_config = MCTSConfig(
        n_simulations=n_simulations,
        UCT_exploration_constant=0.,
        discount_factor=0.99,
        dirichlet_alpha=0.,
        dirichlet_epsilon=0.,
        tau_max=0.,
        tau_min=0.,
        batch_size=32,
        virtual_loss=1.
    )

    replay_config = ReplayConfig()

    actor = Actor(agent, puyo_game, agent_config, mcts_config, replay_config)

    chains = []
    n_games = 10
    for i in tqdm(range(n_games)):
        observation = actor.reset_game()
        root = Node(
            reward=0.,
            done=False,
            agent=actor.agent,
            game=actor.game,
            parent=None,
            config=actor.mcts_config
        )
        done = False
        while not done:
            legal_actions = actor.game.get_legal_actions()
            if with_mcts:
                _, policy, root = run_mcts(actor.agent, actor.game, config=actor.mcts_config, root=root, training=True)
                random_index = random_argmax_in_array(policy[legal_actions])
                action = legal_actions[random_index]
                new_tsumo = [int(p) for p in actor.game.state.queue.queue[2, :]]
                chance_code = get_chance_code(new_tsumo)

                _, reward, done = actor.game.step(action)
                if reward != 0.:
                    if reward == GAMEOVER_REWARD:
                        chains.append(-1)
                    else:
                        chains.append(int(np.round(reward ** (1 / 2.5))))
                else:
                    chains.append(0.)

                try:
                    new_root = root.children[(action, chance_code)]
                    new_root.parent = None
                except KeyError:
                    new_root = Node(
                        reward=reward,
                        done=done,
                        agent=actor.agent,
                        game=actor.game,
                        parent=None,
                        config=actor.mcts_config
                    )

                root = new_root

            else:
                _, policy = actor.agent(observation)
                random_index = random_argmax_in_array(policy[legal_actions])
                action = legal_actions[random_index]

                observation, reward, done = actor.game.step(action)

                if reward != 0.:
                    if reward == GAMEOVER_REWARD:
                        chains.append(-1)
                    else:
                        chains.append(int(np.round(reward ** (1 / 2.5))))
                else:
                    chains.append(0.)

    fig, ax = plt.subplots(figsize=(14, 5))
    bins_edges = np.arange(-1.5, 19.6, 1)  # bords à -1.5, -0.5, 0.5, ..., 19.5
    counts, _ = np.histogram(chains, bins=bins_edges)
    probabilities = counts / counts.sum()
    bin_centers = np.arange(-1, 20, 1)
    ax.bar(bin_centers, probabilities, width=1.0, color="steelblue", edgecolor="white", linewidth=0.8)
    ax.set_xticks(bin_centers)

    ax.set_xlabel("Chain length", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(f'Histogram of chains by {agent_type} during {n_games} games', fontsize=13)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xlim(-2, 20)

    plt.tight_layout()
    plt.savefig("./chain_histogram.png", dpi=150)
