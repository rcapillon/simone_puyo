from dataclasses import dataclass
import numpy as np


@dataclass
class MCTSConfig:
    """
    Dataclass to configure the Monte Carlo Tree Search
    """
    n_simulations: int = 100

    # Number of leaves collected before a single batched network call.
    # Must divide n_simulations evenly for simplicity, but is handled
    # gracefully even if it does not.
    # Rule of thumb: 8–32 is a good starting range. Larger values
    # increase GPU utilisation but delay backpropagation.
    batch_size: int = 8

    UCT_exploration_constant: float = 1.5
    discount_factor: float = 0.99

    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Virtual loss magnitude.
    # A value of 1.0 temporarily counts each in-flight simulation as a
    # loss of -1, steering other simulations away from the same path.
    # Increase if your branching factor is low and collisions are frequent.
    virtual_loss: float = 1.0

    # Temperature schedule based on board fill ratio
    # tau_max  : temperature applied on an empty board (fill = 0)
    # tau_min  : temperature applied on a full board   (fill = 1)
    # The actual temperature is linearly interpolated between the two
    # based on the current board fill ratio.
    # Set tau_max = tau_min = 1.0 to disable (original behaviour).
    tau_max: float = 2.
    tau_min: float = 0.5

    # Nombre de steps pour le bootstrapping des returns.
    # None = Monte Carlo pur (comportement original).
    # 5–10 : bon compromis biais/variance pour des épisodes de ~80 steps.
    n_steps: int = 5

    def __post_init__(self):
        pass


class Node:
    """
    Class for node in the tree search.

    Network evaluation is now *deferred*: nodes are created without
    calling the agent. The agent is called externally in batches by
    _evaluate_batch(), which sets .value, .policy, and .is_evaluated.
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

        # Network outputs - populated lazily by _evaluate_batch().
        # Terminal nodes need no evaluation: value is 0 by convention.
        self.value = 0. if done else None
        self.policy = None
        self.is_evaluated = done   # terminal nodes are considered already evaluated

        self.children = {}

    # ------------------------------------------------------------------
    # Virtual loss helpers
    # ------------------------------------------------------------------

    def apply_virtual_loss(self):
        """
        Temporarily record this node as if it yielded a loss.
        This discourages other in-flight simulations from following the
        same path before backpropagation completes.
        """
        self.N += 1
        self.value_sum -= self.config.virtual_loss

    def undo_virtual_loss(self):
        """
        Revert the effect of apply_virtual_loss() before real
        backpropagation is performed.
        """
        self.N -= 1
        self.value_sum += self.config.virtual_loss

    # ------------------------------------------------------------------
    # Standard MCTS node methods
    # ------------------------------------------------------------------

    def is_done(self):
        return self.done

    def get_value(self):
        if self.N == 0:
            return 0.
        return self.value_sum / self.N

    def calculate_UCT_scores(self):
        """
        Calculates UCT scores per action.

        Q(a) is the average value over all chance_codes visited for that
        action - an unbiased estimate of E_cc[V(s')] without requiring
        explicit chance nodes.  UCT exploration is also aggregated at the
        action level using the total visit count across all chance_codes.

        NOTE: called only on nodes that are already evaluated (is_evaluated=True),
        so self.policy is guaranteed to be set.
        """
        UCT_scores = {}

        for action in self.legal_actions:
            visited_children = [
                child for (a, _), child in self.children.items() if a == action
            ]

            if visited_children:
                Q = np.mean([child.get_value() for child in visited_children])
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
        Returns a specific child node, creating it if inexistant.
        The child is created *without* a network call (is_evaluated=False).
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


# ======================================================================
# Batched network evaluation
# ======================================================================

def _evaluate_batch(nodes):
    """
    Evaluate a list of unevaluated, non-terminal nodes with a single
    batched forward pass.

    Agent interface:
        values, policies = agent(batched_input)

    Where:
        batched_input : np.ndarray of shape (B, *input_shape)
        values        : array-like of shape (B,)  - one scalar per input
        policies      : array-like of shape (B, n_actions)

    Both Keras models and plain numpy functions are supported, as long as
    they accept a numpy batch and return numpy-convertible outputs.

    Keras example:
        # A Keras model with two output heads naturally supports this:
        policy_output, value_output = model(inputs, training=False)
        # np.array() converts Keras tensors transparently.

    Numpy function example:
        values, policies = my_numpy_agent(inputs)
        # Already numpy arrays - nothing special needed.
    """
    if not nodes:
        return

    # Stack game inputs into a single batch along axis 0: (B, *input_shape)
    inputs = np.stack([node.game.get_input() for node in nodes], axis=0)

    values, policies = nodes[0].agent(inputs)

    # np.array() handles both Keras tensors and plain numpy arrays uniformly
    values   = np.array(values).flatten()   # shape (B,)
    policies = np.array(policies)           # shape (B, n_actions)

    for node, value, policy in zip(nodes, values, policies):
        node.value        = float(value)
        node.policy       = policy          # 1-D numpy array of length n_actions
        node.is_evaluated = True


# ======================================================================
# Tree traversal
# ======================================================================

def _select_leaf(root):
    """
    Traverse the tree from *root* following UCT scores until reaching
    either a terminal node or an unevaluated leaf (is_evaluated=False).

    Virtual loss is applied to every node along the path - including the
    leaf - so that concurrent simulations within the same batch are steered
    toward different parts of the tree.

    Returns:
        leaf  : the terminal or unevaluated node at the end of the path
        path  : list of all nodes from root to leaf (inclusive), used to
                undo virtual losses after backpropagation
    """
    node = root
    path = []

    while node.is_evaluated and not node.done:
        node.apply_virtual_loss()
        path.append(node)

        UCT_scores  = node.calculate_UCT_scores()
        action      = max(UCT_scores, key=UCT_scores.__getitem__)
        chance_code = int(np.random.randint(16))
        node        = node.get_or_create_child(action, chance_code)

    # node is now either terminal or unevaluated - apply virtual loss here too
    node.apply_virtual_loss()
    path.append(node)

    return node, path


# ======================================================================
# Backpropagation
# ======================================================================

def backpropagate(node):
    """
    Backpropagates through parent nodes, updating value and visit count.
    Called *after* virtual losses have been undone on the whole path.
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


# ======================================================================
# Utility functions
# ======================================================================

def compute_board_fill_ratio(game):
    """
    Returns the fraction of non-empty cells on the board (rows 1-12, ignoring
    the hidden top row 0 used for game-over detection).
    Board is 13 rows x 6 cols = 78 cells total, playable area = 12 x 6 = 72.
    """
    playable_board = game.state.board.num_board[1:, :]   # rows 1-12, shape (12, 6)
    n_filled = int(np.count_nonzero(playable_board))
    return n_filled / 72.


def temperature_from_fill(fill_ratio, tau_max, tau_min):
    """
    Linearly interpolates temperature between tau_max (empty board) and
    tau_min (full board) based on current fill ratio.

        tau = tau_max - fill_ratio * (tau_max - tau_min)

    Examples with tau_max=2.0, tau_min=0.5:
        fill=0.00  -> tau=2.00  (very diverse policy, board empty)
        fill=0.25  -> tau=1.625
        fill=0.50  -> tau=1.25
        fill=0.75  -> tau=0.875
        fill=1.00  -> tau=0.50  (sharper policy, board dense)
    """
    return tau_max - fill_ratio * (tau_max - tau_min)


def apply_temperature(visit_counts, temperature):
    """
    Converts raw visit counts to a policy using a temperature parameter.

        policy[a]  proportional to  N[a] ^ (1 / temperature)

    Special cases:
        temperature -> 0  :  deterministic argmax (one-hot on most-visited action)
        temperature = 1   :  proportional to visit counts (original behaviour)
        temperature > 1   :  flatter distribution, more diverse
    """
    if temperature < 1e-6:
        policy = np.zeros_like(visit_counts, dtype=np.float64)
        policy[np.argmax(visit_counts)] = 1.0
        return policy

    counts_temp = visit_counts ** (1.0 / temperature)
    total = counts_temp.sum()
    if total == 0:
        mask   = visit_counts > 0
        policy = mask.astype(np.float64) / mask.sum()
        return policy

    return counts_temp / total


# ======================================================================
# Main entry point
# ======================================================================

def run_mcts(agent, game, config=MCTSConfig(), root=None, training=True):
    """
    Perform n_simulations MCTS simulations using batched network evaluation
    and virtual losses, returning the root value and an MCTS-derived policy.

    Simulation loop
    ---------------
    Each iteration processes a batch of `config.batch_size` simulations:
      1. Select   - traverse the tree to collect `batch_size` leaf nodes,
                    applying virtual loss along each path.
      2. Evaluate - call the network once for all unevaluated leaves.
      3. Undo & Backprop - revert virtual losses, then backpropagate real values.

    Agent interface
    ---------------
    The agent must accept a numpy batch and return two numpy-convertible arrays:

        values, policies = agent(inputs)
            inputs   : np.ndarray  (B, *input_shape)
            values   : array-like  (B,)
            policies : array-like  (B, n_actions)

    Works transparently with:
      - A Keras model called as model(inputs, training=False) wrapped in a lambda.
      - Any plain numpy callable.

    Keras wrapping example:
        agent = lambda x: model(x, training=False)
        value, policy, root = run_mcts(agent, game, config)
    """
    # ------------------------------------------------------------------ root
    if root is None:
        root = Node(
            reward=0.,
            done=False,
            agent=agent,
            game=game,
            parent=None,
            config=config
        )

    # Evaluate root before anything else (batch of 1)
    if not root.is_evaluated:
        _evaluate_batch([root])

    # Dirichlet noise on root policy (training only)
    if root.policy is not None and training:
        noise = np.random.dirichlet([config.dirichlet_alpha] * len(root.legal_actions))
        for i, action in enumerate(root.legal_actions):
            root.policy[action] = (
                (1 - config.dirichlet_epsilon) * root.policy[action]
                + config.dirichlet_epsilon * noise[i]
            )

    # ------------------------------------------------------------------ main loop
    sims_done = 0
    while sims_done < config.n_simulations:
        current_batch = min(config.batch_size, config.n_simulations - sims_done)

        # --- Step 1: selection (with virtual losses) ---
        leaves = []
        paths  = []
        for _ in range(current_batch):
            leaf, path = _select_leaf(root)
            leaves.append(leaf)
            paths.append(path)

        # --- Step 2: batched network evaluation ---
        # Deduplicate by node id: if two paths land on the same unevaluated
        # node we only evaluate it once, but backpropagate twice (correct).
        unevaluated = list(
            {id(leaf): leaf
             for leaf in leaves
             if not leaf.is_evaluated}.values()
        )
        _evaluate_batch(unevaluated)

        # --- Step 3: undo virtual losses, then backpropagate ---
        for leaf, path in zip(leaves, paths):
            for node in path:
                node.undo_virtual_loss()
            backpropagate(leaf)

        sims_done += current_batch

    # ------------------------------------------------------------------ policy
    visit_counts = np.zeros((22,))
    for k, v in root.children.items():
        visit_counts[k[0]] += v.N

    fill_ratio = compute_board_fill_ratio(game)
    tau        = temperature_from_fill(fill_ratio, config.tau_max, config.tau_min)
    policy     = apply_temperature(visit_counts, tau)
    value      = root.get_value()

    return value, policy, root