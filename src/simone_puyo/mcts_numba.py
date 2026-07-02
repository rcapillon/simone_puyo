from dataclasses import dataclass
import numpy as np
import math
import random
from numba import njit


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

    def __post_init__(self):
        pass


@njit(cache=True)
def _select_best_action_numba(legal_actions_arr, child_actions, child_rewards,
                               child_N, child_value_sum, n_children,
                               policy, N_root, discount_factor, exploration_c):
    """
    Selectionne l'action de meilleur score UCT en une passe sur les flat arrays
    des enfants. Appele a chaque etape de traversee dans _select_leaf.

    legal_actions_arr : int32[n_legal]
    child_actions     : int32[n_children]  -- action de chaque enfant cree
    child_rewards     : float64[n_children] -- recompense immediate (fixe)
    child_N           : int64[n_children]   -- N courant (synchro backprop+VL)
    child_value_sum   : float64[n_children] -- value_sum courant (idem)
    n_children        : int                 -- enfants crees jusqu'ici
    policy            : float32[22]         -- prior du reseau (masque+normalise)
    N_root            : int                 -- N du noeud appelant (apres VL)
    discount_factor   : float
    exploration_c     : float
    """
    sqrt_N = math.sqrt(float(N_root))
    best_action = -1
    best_score = -1e18

    for ai in range(legal_actions_arr.shape[0]):
        action = legal_actions_arr[ai]
        total = 0.0
        n_action = 0
        count = 0
        for ci in range(n_children):
            if child_actions[ci] == action:
                cN = child_N[ci]
                child_v = child_value_sum[ci] / cN if cN > 0 else 0.0
                total += child_rewards[ci] + discount_factor * child_v
                n_action += cN
                count += 1
        Q = total / count if count > 0 else 0.0
        U = exploration_c * policy[action] * sqrt_N / (n_action + 1)
        score = Q + U
        if score > best_score:
            best_score = score
            best_action = action

    return best_action


_INITIAL_CHILD_CAP = 8   # capacite initiale des flat arrays par noeud ; double a la demande


class Node:
    """
    Class for node in the tree search.

    Chaque noeud maintient des flat arrays numpy (child_actions, child_rewards,
    child_N, child_value_sum) synchonises en permanence avec les attributs N et
    value_sum de ses enfants. Cela permet de compiler select_best_action en Numba
    sans toucher aux objets Python pendant la selection UCT.

    Invariant : pour tout enfant `c` de ce noeud situe au slot `s` :
        self._child_N[s]          == c.N
        self._child_value_sum[s]  == c.value_sum

    Cet invariant est maintenu par apply_virtual_loss, undo_virtual_loss et
    backpropagate, qui mettent a jour simultanement l'enfant et le slot du parent.
    """
    def __init__(self, reward, done, agent, game, parent=None, config=MCTSConfig()):
        self.config = config
        self.agent = agent
        self.game = game
        self.legal_actions = self.game.get_legal_actions()
        self._legal_actions_arr = np.array(self.legal_actions, dtype=np.int32)
        self.N = 0
        self.value_sum = 0.
        self.reward = reward
        self.parent = parent
        self.done = done
        self.value = 0. if done else None
        self.policy = None
        self.is_evaluated = done
        self.children = {}   # (action, chance_code) -> Node ; conserve pour tree reuse

        # Flat arrays pour les statistiques des enfants directs.
        # Indices valides : 0 .. _n_children-1
        cap = _INITIAL_CHILD_CAP
        self._child_actions = np.empty(cap, dtype=np.int32)
        self._child_rewards = np.empty(cap, dtype=np.float64)
        self._child_N = np.zeros(cap, dtype=np.int64)
        self._child_value_sum = np.zeros(cap, dtype=np.float64)
        self._n_children = 0

        # Index de ce noeud dans les flat arrays de son parent (-1 pour la racine)
        self._parent_slot = -1

    # ------------------------------------------------------------------
    # Gestion de la capacite des flat arrays
    # ------------------------------------------------------------------

    def _grow_child_arrays(self):
        old_cap = len(self._child_actions)
        new_cap = old_cap * 2
        new_a = np.empty(new_cap, dtype=np.int32)
        new_r = np.empty(new_cap, dtype=np.float64)
        new_N = np.zeros(new_cap, dtype=np.int64)
        new_vs = np.zeros(new_cap, dtype=np.float64)
        new_a [:old_cap] = self._child_actions
        new_r [:old_cap] = self._child_rewards
        new_N [:old_cap] = self._child_N
        new_vs[:old_cap] = self._child_value_sum
        self._child_actions = new_a
        self._child_rewards = new_r
        self._child_N = new_N
        self._child_value_sum = new_vs

    # ------------------------------------------------------------------
    # Virtual loss
    # ------------------------------------------------------------------

    def apply_virtual_loss(self):
        """
        Incremente N et decremente value_sum pour decourager les autres
        simulations du meme batch de prendre ce chemin avant le backprop reel.
        Synchonise simultanement le slot correspondant dans les flat arrays du parent.
        """
        self.N          += 1
        self.value_sum  -= self.config.virtual_loss
        if self.parent is not None:
            self.parent._child_N[self._parent_slot] += 1
            self.parent._child_value_sum[self._parent_slot] -= self.config.virtual_loss

    def undo_virtual_loss(self):
        """Inverse de apply_virtual_loss, appele juste avant backpropagate."""
        self.N          -= 1
        self.value_sum  += self.config.virtual_loss
        if self.parent is not None:
            self.parent._child_N[self._parent_slot] -= 1
            self.parent._child_value_sum[self._parent_slot] += self.config.virtual_loss

    # ------------------------------------------------------------------
    # Methodes MCTS standard
    # ------------------------------------------------------------------

    def is_done(self):
        return self.done

    def get_value(self):
        if self.N == 0:
            return 0.
        return self.value_sum / self.N

    def select_best_action(self):
        """
        Retourne l'action de meilleur score UCT via la fonction Numba compilee.
        Appelee uniquement sur des noeuds deja evalues (policy garanti non None).
        """
        return int(_select_best_action_numba(
            self._legal_actions_arr,
            self._child_actions[:self._n_children],
            self._child_rewards[:self._n_children],
            self._child_N[:self._n_children],
            self._child_value_sum[:self._n_children],
            self._n_children,
            self.policy,
            self.N,
            self.config.discount_factor,
            self.config.UCT_exploration_constant
        ))

    def get_or_create_child(self, action, chance_code):
        """
        Retourne le noeud enfant correspondant, en le creant si necessaire.
        A la creation : alloue un slot dans les flat arrays du parent et fixe
        le _parent_slot de l'enfant pour que les synchros ulterieures soient O(1).
        """
        key = (action, chance_code)
        if key not in self.children:
            new_game = self.game.copy()
            _, reward, done = new_game.step(action)
            new_game.state.queue.insert_last_in_queue(chance_code)

            new_child = Node(
                reward=reward,
                done=done,
                agent=self.agent,
                game=new_game,
                parent=self,
                config=self.config
            )

            # Alloue le slot dans les flat arrays
            slot = self._n_children
            if slot >= len(self._child_actions):
                self._grow_child_arrays()

            self._child_actions   [slot] = action
            self._child_rewards   [slot] = float(reward)
            self._child_N         [slot] = 0
            self._child_value_sum [slot] = 0.0
            self._n_children = slot + 1
            new_child._parent_slot = slot

            self.children[key] = new_child

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
        legal = node.legal_actions
        masked_policy = np.zeros_like(policy)
        masked_policy[legal] = policy[legal]
        total = masked_policy.sum()
        if total > 1e-8:
            masked_policy /= total
        else:
            masked_policy[legal] = 1.0 / len(legal)
        node.value = float(value)
        node.policy = masked_policy
        node.is_evaluated = True


# ======================================================================
# Tree traversal
# ======================================================================

# def _select_leaf(root):
#     """
#     Traverse the tree from *root* following UCT scores until reaching
#     either a terminal node or an unevaluated leaf (is_evaluated=False).
#
#     Virtual loss is applied to every node along the path - including the
#     leaf - so that concurrent simulations within the same batch are steered
#     toward different parts of the tree.
#
#     Returns:
#         leaf  : the terminal or unevaluated node at the end of the path
#         path  : list of all nodes from root to leaf (inclusive), used to
#                 undo virtual losses after backpropagation
#     """
#     node = root
#     path = []
#
#     while node.is_evaluated and not node.done:
#         node.apply_virtual_loss()
#         path.append(node)
#
#         UCT_scores  = node.calculate_UCT_scores()
#         action      = max(UCT_scores, key=UCT_scores.__getitem__)
#         chance_code = int(np.random.randint(16))
#         node        = node.get_or_create_child(action, chance_code)
#
#     # node is now either terminal or unevaluated - apply virtual loss here too
#     node.apply_virtual_loss()
#     path.append(node)
#
#     return node, path


def _select_leaf(root):
    """
    Traverse the tree from *root* following UCT scores until reaching
    either a terminal node or an unevaluated leaf (is_evaluated=False).
    """
    node = root
    path = []

    while node.is_evaluated and not node.done:
        node.apply_virtual_loss()
        path.append(node)

        action      = node.select_best_action()
        chance_code = random.randint(0, 15)
        node        = node.get_or_create_child(action, chance_code)

    node.apply_virtual_loss()
    path.append(node)

    return node, path


# ======================================================================
# Backpropagation
# ======================================================================

def backpropagate(node):
    """
    Remonte la valeur depuis la feuille jusqu'a la racine.
    A chaque noeud, met a jour simultanement :
      - node.value_sum et node.N (comme avant)
      - parent._child_value_sum[slot] et parent._child_N[slot] (nouveau)

    Invariant maintenu : parent._child_X[slot] == node.X apres chaque backprop.
    """
    value = node.value
    while node is not None:
        node.value_sum += value
        node.N += 1
        parent = node.parent
        if parent is not None:
            parent._child_N[node._parent_slot] += 1
            parent._child_value_sum[node._parent_slot] += value
            value = node.reward + node.config.discount_factor * value
        node = parent


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
