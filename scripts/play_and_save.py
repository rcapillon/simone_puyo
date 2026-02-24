"""
play_and_save.py
================
Joue une partie complète de Puyo Puyo avec MCTS et sauvegarde toutes les données
utiles pour l'analyse post-partie (board states, policies, values, rewards…).

Usage:
    python play_and_save.py [--agent mlp|resnet] [--model-dir ../saved_agents]
                            [--simulations 200] [--max-moves 50] [--out ./game_replay.pkl]
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional

from src.simone_puyo.agents import ResNetConfig, ResNetAgent, MLPConfig, MLPAgent
from src.simone_puyo.puyo import PuyoGame, GAMEOVER_REWARD, reward_dict
from src.simone_puyo.mcts import MCTSConfig, Node, run_mcts
from src.simone_puyo.utils import random_argmax_in_array
from src.simone_puyo.puyo import get_chance_code


# ---------------------------------------------------------------------------
# Data structures for a replay
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """All data captured at a single game step."""
    step: int

    # Board state (numeric 13×6 int array) BEFORE the move
    board_num: np.ndarray

    # Queue state (3×2 int array) BEFORE the move
    queue_num: np.ndarray

    # One-hot observation fed to the network (14×6×5)
    observation: np.ndarray

    # Legal actions available
    legal_actions: List[int]

    # MCTS outputs
    mcts_policy: np.ndarray       # policy vector (22,) from visit counts
    mcts_value: float             # root value estimate from MCTS
    network_value: float          # raw network value (before MCTS)
    network_policy: np.ndarray    # raw network policy (before MCTS)

    # Chosen action and outcome
    action: int
    reward: float
    chain_length: int             # 0 if no chain
    gameover: bool

    # UCT score distribution for the chosen step (dict keyed by (action, chance_code))
    uct_scores: Optional[dict] = None


@dataclass
class GameReplay:
    """Full replay of one Puyo game."""
    agent_name: str
    agent_type: str                       # 'mlp' or 'resnet'
    mcts_config: MCTSConfig
    max_moves: int
    steps: List[StepRecord] = field(default_factory=list)
    total_reward: float = 0.0
    n_steps: int = 0
    gameover: bool = False
    discounted_returns: List[float] = field(default_factory=list)
    discount_factor: float = 0.99

    def compute_returns(self):
        rewards = [s.reward for s in self.steps]
        value = 0.0
        returns = []
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1 and rewards[i] == GAMEOVER_REWARD:
                value = rewards[i]
            else:
                value = rewards[i] + self.discount_factor * value
            returns.insert(0, value)
        self.discounted_returns = returns


# ---------------------------------------------------------------------------
# Core play function
# ---------------------------------------------------------------------------

def play_game_with_mcts(agent, game: PuyoGame, mcts_config: MCTSConfig) -> GameReplay:
    """
    Play a full game, recording everything needed for post-game analysis.
    Returns a GameReplay object.
    """
    replay = GameReplay(
        agent_name=agent.name,
        agent_type=type(agent).__name__,
        mcts_config=mcts_config,
        max_moves=game.max_moves,
        discount_factor=mcts_config.discount_factor,
    )

    observation = game.reset()

    root = Node(
        reward=0.,
        done=False,
        agent=agent,
        game=game,
        parent=None,
        config=mcts_config,
    )

    step = 0
    done = False

    while not done:
        # Capture pre-move state
        board_snapshot = game.state.board.num_board.copy()
        queue_snapshot = game.state.queue.queue.copy()
        legal_actions = game.get_legal_actions()

        # Raw network outputs (before MCTS noise)
        raw_value, raw_policy = agent(observation)

        # Run MCTS
        mcts_value, mcts_policy, root = run_mcts(
            agent, game, config=mcts_config, root=root, training=False
        )

        # Capture UCT scores at root for analysis
        try:
            uct_scores = root.calculate_UCT_scores()
            # Keep only actions that are legal for clarity
            uct_legal = {k: v for k, v in uct_scores.items() if k[0] in legal_actions}
        except Exception:
            uct_legal = {}

        # Select action
        random_index = random_argmax_in_array(mcts_policy[legal_actions])
        action = legal_actions[random_index]

        # Determine next tsumo chance code (for tree reuse)
        new_tsumo = [int(p) for p in game.state.queue.queue[2, :]]
        chance_code = get_chance_code(new_tsumo)

        # Step the environment
        next_observation, reward, done = game.step(action)
        gameover = game.state.board.check_gameover()

        # Infer chain length from reward (reverse of reward_dict)
        if reward == GAMEOVER_REWARD:
            chain_length = 0
        else:
            chain_length = 0
            for k, v in reward_dict.items():
                if abs(v - reward) < 1e-6:
                    chain_length = k
                    break

        record = StepRecord(
            step=step,
            board_num=board_snapshot,
            queue_num=queue_snapshot,
            observation=observation.copy(),
            legal_actions=legal_actions,
            mcts_policy=mcts_policy.copy(),
            mcts_value=float(mcts_value),
            network_value=float(raw_value) if np.isscalar(raw_value) else float(raw_value.flatten()[0]),
            network_policy=raw_policy.copy() if raw_policy.ndim == 1 else raw_policy[0].copy(),
            action=action,
            reward=float(reward),
            chain_length=chain_length,
            gameover=gameover,
            uct_scores=uct_legal,
        )
        replay.steps.append(record)
        replay.total_reward += reward

        # Advance root node
        try:
            root = root.children[(action, chance_code)]
            root.parent = None
        except KeyError:
            root = Node(
                reward=reward,
                done=done,
                agent=agent,
                game=game,
                parent=None,
                config=mcts_config,
            )

        observation = next_observation
        step += 1

    replay.n_steps = step
    replay.gameover = done and game.state.board.check_gameover()
    replay.compute_returns()

    return replay


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Play one Puyo game with MCTS and save replay.')
    parser.add_argument('--agent', choices=['mlp', 'resnet'], default='mlp')
    parser.add_argument('--model-dir', default='../saved_agents',
                        help='Directory containing saved agent files')
    parser.add_argument('--simulations', type=int, default=200,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--uct-c', type=float, default=1.5,
                        help='UCT exploration constant')
    parser.add_argument('--discount', type=float, default=0.99,
                        help='Discount factor for MCTS / returns')
    parser.add_argument('--dirichlet-alpha', type=float, default=0.3)
    parser.add_argument('--dirichlet-epsilon', type=float, default=0.25)
    parser.add_argument('--max-moves', type=int, default=50,
                        help='Maximum number of moves per game')
    parser.add_argument('--out', default='./game_replay.pkl',
                        help='Output path for the replay pickle file')
    parser.add_argument('--no-mcts', action='store_true',
                        help='Skip MCTS, use raw network policy only (fast)')
    return parser.parse_args()


def build_agent(agent_type: str, model_dir: str):
    if agent_type == 'mlp':
        config = MLPConfig()
        agent = MLPAgent(name='mlp_agent_1', config=config)
    else:
        config = ResNetConfig()
        agent = ResNetAgent(name='resnet_agent_1', config=config)

    if model_dir and os.path.isdir(model_dir):
        print(f'Loading agent from {model_dir}…')
        agent.load_model(model_dir)
    else:
        print('No model directory found — building a fresh (untrained) model.')
        agent.build_model()

    return agent, config


def main():
    args = parse_args()

    # Build agent
    agent, agent_config = build_agent(args.agent, args.model_dir)
    print(f'Agent ready: {agent.name}')

    # MCTS config
    mcts_config = MCTSConfig(
        n_simulations=args.simulations,
        UCT_exploration_constant=args.uct_c,
        discount_factor=args.discount,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
    )

    # Game
    game = PuyoGame(max_moves=args.max_moves)
    print(f'Playing game (max_moves={args.max_moves}, simulations={args.simulations})…')

    if args.no_mcts:
        # Simple greedy play without MCTS for speed testing
        mcts_config_light = MCTSConfig(n_simulations=1)
        replay = play_game_with_mcts(agent, game, mcts_config_light)
    else:
        replay = play_game_with_mcts(agent, game, mcts_config)

    # Summary
    print(f'\n=== Game Over ===')
    print(f'Steps played  : {replay.n_steps}')
    print(f'Total reward  : {replay.total_reward:.2f}')
    print(f'Game over     : {replay.gameover}')
    chains = [s.chain_length for s in replay.steps]
    print(f'Chains per step: max={max(chains)}, mean={np.mean(chains):.2f}')
    print(f'Moves with chain: {sum(c > 0 for c in chains)}/{replay.n_steps}')

    # Save
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(replay, f)
    print(f'\nReplay saved → {out_path}')


if __name__ == '__main__':
    main()