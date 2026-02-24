from dataclasses import dataclass
import numpy as np

from .utils import random_max_in_dict


@dataclass
class MCTSConfig:
    """
    Dataclass to configure the Monte Carlo Tree Search
    """
    n_simulations: int = 100

    UCT_exploration_constant: float = 1.5
    discount_factor: float = 0.99

    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Temperature schedule based on board fill ratio
    # tau_max  : temperature applied on an empty board (fill = 0)
    # tau_min  : temperature applied on a full board   (fill = 1)
    # The actual temperature is linearly interpolated between the two
    # based on the current board fill ratio.
    # Set tau_max = tau_min = 1.0 to disable (original behaviour).
    tau_max: float = 2.
    tau_min: float = 0.5

    def __post_init__(self):
        pass


class Node:
    """
    Class for node in the tree search
    """
    def __init__(self, reward, done, agent, game, parent=None, config=MCTSConfig()):
        self.config = config

        self.agent = agent
        self.game = game

        self.legal_actions = self.game.get_legal_actions()

        self.N = 0
        self.value_sum = 0.
        self.reward = reward

        self.parent = parent
        self.done = done
        if not done:
            self.value, self.policy = self.agent(self.game.get_input())
        else:
            self.value, self.policy = 0., None

        self.children = {}

    def is_done(self):
        """
        returns completion status of the game
        """
        return True if self.done else False

    def get_value(self):
        """
        returns current node value
        """
        if self.N == 0:
            return 0.
        else:
            return self.value_sum / self.N

    def calculate_UCT_scores(self):
        """
        Calculates UCT scores per action.

        Q(a) is the average value over all chance_codes visited for that
        action — an unbiased estimate of E_cc[V(s')] without requiring
        explicit chance nodes.  UCT exploration is also aggregated at the
        action level using the total visit count across all chance_codes.
        """
        UCT_scores = {}

        for action in self.legal_actions:
            # Collect all children reached via this action
            visited_children = [
                child for (a, _), child in self.children.items() if a == action
            ]

            if visited_children:
                # Q = average value over observed chance outcomes
                Q = np.mean([child.get_value() for child in visited_children])
                # N_action = total visits through this action (all chance_codes)
                N_action = sum(child.N for child in visited_children)
            else:
                Q = 0.
                N_action = 0

            U = (self.config.UCT_exploration_constant
                 * self.policy[action]
                 * np.sqrt(self.N) / (N_action + 1))

            UCT_scores[action] = Q + U

        return UCT_scores

    def get_or_create_child(self, action, chance_code):
        """
        returns a specific child node, creating it if inexistant
        """
        key = (action, chance_code)
        if key not in self.children:
            new_game = self.game.copy()
            _, reward, done = new_game.step(action)
            new_game.state.queue.insert_last_in_queue(chance_code)

            self.children[key] = Node(
                reward=reward,
                done=done,
                agent=self.agent,
                game=new_game,
                parent=self,
                config=self.config
            )

        return self.children[key]


def backpropagate(node):
    """
    backpropagates through parent nodes, updating value and visit count
    """
    value = node.reward + node.value
    node.value_sum += value
    node.N += 1

    parent = node.parent
    while parent is not None:
        value = parent.reward + parent.config.discount_factor * value
        parent.value_sum += value
        parent.N += 1
        parent = parent.parent


def compute_board_fill_ratio(game):
    """
    Returns the fraction of non-empty cells on the board (rows 1-12, ignoring
    the hidden top row 0 used for game-over detection).
    Board is 13 rows × 6 cols = 78 cells total, playable area = 12 × 6 = 72.
    """
    playable_board = game.state.board.num_board[1:, :]   # rows 1–12, shape (12, 6)
    n_filled = int(np.count_nonzero(playable_board))
    return n_filled / 72.


def temperature_from_fill(fill_ratio, tau_max, tau_min):
    """
    Linearly interpolates temperature between tau_max (empty board) and
    tau_min (full board) based on current fill ratio.

        tau = tau_max - fill_ratio * (tau_max - tau_min)

    Examples with tau_max=2.0, tau_min=0.5:
        fill=0.00  → tau=2.00  (very diverse policy, board empty)
        fill=0.25  → tau=1.625
        fill=0.50  → tau=1.25
        fill=0.75  → tau=0.875
        fill=1.00  → tau=0.50  (sharper policy, board dense)
    """
    return tau_max - fill_ratio * (tau_max - tau_min)


def apply_temperature(visit_counts, temperature):
    """
    Converts raw visit counts to a policy using a temperature parameter.

        policy[a]  ∝  N[a] ^ (1 / temperature)

    Special cases:
        temperature → 0  :  deterministic argmax (one-hot on most-visited action)
        temperature = 1  :  proportional to visit counts (original behaviour)
        temperature > 1  :  flatter distribution, more diverse
    """
    if temperature < 1e-6:
        # Greedy: one-hot on the most visited action
        policy = np.zeros_like(visit_counts, dtype=np.float64)
        policy[np.argmax(visit_counts)] = 1.0
        return policy

    counts_temp = visit_counts ** (1.0 / temperature)
    total = counts_temp.sum()
    if total == 0:
        # Fallback: uniform over visited actions
        mask = visit_counts > 0
        policy = mask.astype(np.float64) / mask.sum()
        return policy

    return counts_temp / total


def run_mcts(agent, game, config=MCTSConfig(), root=None, training=True):
    """
    Perform multiple MCTS simulations, returning root node value and MCTS-based policy.

    The output policy is computed from visit counts with a temperature that
    decreases as the board fills up (config.tau_max → config.tau_min).
    Set tau_max = tau_min = 1.0 to recover the original behaviour.
    """
    if root is None:
        root = Node(
            reward=0.,
            done=False,
            agent=agent,
            game=game,
            parent=None,
            config=config
        )

    if root.policy is not None and training:
        noise = np.random.dirichlet([config.dirichlet_alpha] * len(root.legal_actions))
        for i, action in enumerate(root.legal_actions):
            root.policy[action] = (
                    (1 - config.dirichlet_epsilon) * root.policy[action]
                    + config.dirichlet_epsilon * noise[i]
            )

    for _ in range(config.n_simulations):
        node = root
        is_leaf = False
        while not is_leaf:
            if not node.is_done():
                UCT_scores = node.calculate_UCT_scores()
                # UCT_scores is now keyed by action only
                action = max(UCT_scores, key=UCT_scores.__getitem__)
                # Draw chance_code uniformly
                chance_code = int(np.random.randint(16))
                node = node.get_or_create_child(action, chance_code)
                if node.N == 0:
                    is_leaf = True
            else:
                is_leaf = True
        backpropagate(node)

    # Aggregate visit counts across all children
    visit_counts = np.zeros((22,))
    for k, v in root.children.items():
        visit_counts[k[0]] += v.N

    # Compute temperature from current board fill ratio
    fill_ratio = compute_board_fill_ratio(game)
    tau = temperature_from_fill(fill_ratio, config.tau_max, config.tau_min)

    policy = apply_temperature(visit_counts, tau)

    value = root.get_value()
    return value, policy, root