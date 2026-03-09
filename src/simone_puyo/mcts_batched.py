from dataclasses import dataclass
import numpy as np


@dataclass
class MCTSConfig:
    """
    Dataclass to configure the Monte Carlo Tree Search
    """
    n_simulations: int = 100

    # Number of leaves collected before a single batched network call.
    # Rule of thumb: batch_size <= n_simulations / 4 to guarantee at least
    # 4 rounds of backpropagation. With n_simulations=1000, max is ~250,
    # recommended is 128 (8 rounds — good GPU utilisation / tree quality balance).
    batch_size: int = 8

    UCT_exploration_constant: float = 1.5
    discount_factor: float = 0.99

    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Virtual loss magnitude.
    # A value of 1.0 temporarily counts each in-flight simulation as a
    # loss of -1, steering other simulations away from the same path.
    virtual_loss: float = 1.0

    # Temperature schedule based on board fill ratio.
    # tau_max : temperature applied on an empty board (fill = 0)
    # tau_min : temperature applied on a full board   (fill = 1)
    # Linearly interpolated based on current board fill ratio.
    tau_max: float = 2.
    tau_min: float = 0.5

    # Nombre de steps pour le bootstrapping des returns.
    # None = Monte Carlo pur (comportement original).
    # Pour des épisodes de 20 steps, recommandé : 15–18 avec curriculum.
    n_steps: int = 5

    def __post_init__(self):
        pass


class Node:
    """
    Class for node in the tree search.

    Network evaluation is *deferred*: nodes are created without calling the
    agent. The agent is called externally in batches by _evaluate_batch(),
    which sets .value, .policy, and .is_evaluated.

    Optimisation : l'input one-hot est calculé une seule fois à la création
    du node et mis en cache dans _cached_input. Cela évite de recalculer
    get_input() lors de chaque appel à _evaluate_batch().
    """
    def __init__(self, reward, done, agent, game, parent=None, config=MCTSConfig()):
        self.config = config

        self.agent = agent
        self.game  = game

        self.legal_actions = self.game.get_legal_actions()

        self.N          = 0
        self.value_sum  = 0.
        self.reward     = reward

        self.parent = parent
        self.done   = done

        # Network outputs - populated lazily by _evaluate_batch().
        # Terminal nodes need no evaluation: value is 0 by convention.
        self.value        = 0. if done else None
        self.policy       = None
        self.is_evaluated = done  # terminal nodes are considered already evaluated

        self.children = {}

        # Cache de l'input one-hot calculé à la création du node.
        # Un node est évalué au plus une fois → calculer get_input() une seule
        # fois à la création et le réutiliser dans _evaluate_batch() et
        # dans le bootstrap de collect_game() élimine des recalculs redondants.
        self._cached_input = game.get_input()

    # ------------------------------------------------------------------
    # Virtual loss helpers
    # ------------------------------------------------------------------

    def apply_virtual_loss(self):
        """
        Temporarily record this node as if it yielded a loss.
        Discourages other in-flight simulations from following the same path.
        """
        self.N          += 1
        self.value_sum  -= self.config.virtual_loss

    def undo_virtual_loss(self):
        """
        Revert the effect of apply_virtual_loss() before real backpropagation.
        """
        self.N          -= 1
        self.value_sum  += self.config.virtual_loss

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
        action — an unbiased estimate of E_cc[V(s')] without explicit chance
        nodes. UCT exploration is aggregated at the action level.

        Called only on nodes that are already evaluated (is_evaluated=True),
        so self.policy is guaranteed to be set.
        """
        UCT_scores = {}

        for action in self.legal_actions:
            visited_children = [
                child for (a, _), child in self.children.items() if a == action
            ]

            if visited_children:
                Q        = np.mean([child.get_value() for child in visited_children])
                N_action = sum(child.N for child in visited_children)
            else:
                Q        = 0.
                N_action = 0

            U = (self.config.UCT_exploration_constant
                 * self.policy[action]
                 * np.sqrt(self.N) / (N_action + 1))

            UCT_scores[action] = Q + U

        return UCT_scores

    def get_or_create_child(self, action, chance_code):
        """
        Returns a specific child node, creating it if it does not exist.
        The child is created without a network call (is_evaluated=False).
        The one-hot input is cached on creation via _cached_input.
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

    Utilise node._cached_input au lieu de recalculer node.game.get_input()
    — l'input a été calculé une fois à la création du node et mis en cache.

    Agent interface:
        values, policies = agent(batched_input)
        batched_input : np.ndarray of shape (B, *input_shape)
        values        : array-like of shape (B,)
        policies      : array-like of shape (B, n_actions)
    """
    if not nodes:
        return

    # Utilise le cache : np.stack sur des arrays déjà calculés,
    # pas d'appel à game.get_input() ici
    inputs = np.stack([node._cached_input for node in nodes], axis=0)

    values, policies = nodes[0].agent(inputs)

    values   = np.array(values).flatten()  # shape (B,)
    policies = np.array(policies)          # shape (B, n_actions)

    for node, value, policy in zip(nodes, values, policies):
        node.value        = float(value)
        node.policy       = policy
        node.is_evaluated = True


# ======================================================================
# Tree traversal
# ======================================================================

def _select_leaf(root):
    """
    Traverse the tree from root following UCT scores until reaching
    either a terminal node or an unevaluated leaf (is_evaluated=False).

    Virtual loss is applied along the entire path including the leaf,
    steering concurrent simulations toward different parts of the tree.

    Returns:
        leaf : terminal or unevaluated node at the end of the path
        path : list of all nodes from root to leaf (inclusive)
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

    node.apply_virtual_loss()
    path.append(node)

    return node, path


# ======================================================================
# Backpropagation
# ======================================================================

def backpropagate(node):
    """
    Backpropagates through parent nodes, updating value and visit count.
    Called after virtual losses have been undone on the whole path.
    """
    value = node.reward + node.value
    node.value_sum += value
    node.N         += 1

    parent = node.parent
    while parent is not None:
        value           = parent.reward + parent.config.discount_factor * value
        parent.value_sum += value
        parent.N        += 1
        parent = parent.parent


# ======================================================================
# Utility functions
# ======================================================================

def compute_board_fill_ratio(game):
    """
    Returns the fraction of non-empty cells on the playable board (rows 1-12).
    Playable area = 12 x 6 = 72 cells.
    """
    playable_board = game.state.board.num_board[1:, :]
    n_filled = int(np.count_nonzero(playable_board))
    return n_filled / 72.


def temperature_from_fill(fill_ratio, tau_max, tau_min):
    """
    Linearly interpolates temperature between tau_max (empty) and tau_min (full).

        tau = tau_max - fill_ratio * (tau_max - tau_min)
    """
    return tau_max - fill_ratio * (tau_max - tau_min)


def apply_temperature(visit_counts, temperature):
    """
    Converts raw visit counts to a policy using a temperature parameter.

        policy[a]  proportional to  N[a] ^ (1 / temperature)

    Special cases:
        temperature -> 0 : deterministic argmax
        temperature = 1  : proportional to visit counts
        temperature > 1  : flatter, more diverse distribution
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
    Each iteration processes a batch of config.batch_size simulations:
      1. Select   - traverse the tree to collect batch_size leaf nodes,
                    applying virtual loss along each path.
      2. Evaluate - call the network once for all unevaluated leaves,
                    using their _cached_input (no recomputation).
      3. Undo & Backprop - revert virtual losses, then backpropagate.
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

        # Step 1 : selection
        leaves = []
        paths  = []
        for _ in range(current_batch):
            leaf, path = _select_leaf(root)
            leaves.append(leaf)
            paths.append(path)

        # Step 2 : batched evaluation (deduplication par node id)
        unevaluated = list(
            {id(leaf): leaf
             for leaf in leaves
             if not leaf.is_evaluated}.values()
        )
        _evaluate_batch(unevaluated)

        # Step 3 : undo virtual losses + backpropagation
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