from dataclasses import dataclass
import numpy as np

from .utils import random_max_in_dict


@dataclass
class MCTSConfig:
    """
    Dataclass to configure the Monte Carlo Tree Search
    """
    n_simulations: int = 100

    UCT_exploration_constant: float = 4.
    discount_factor: float = 0.97

    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    base_temperature: float = 1.0  # τ=1 exploration, τ→0 greedy

    def __post_init__(self):
        if self.base_temperature <= 0.1:
            self.base_temperature = 0.1


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
        calculates UCT scores (Upper Confidence bound applied to Trees) for all possible children nodes
        """
        UCT_scores = {}

        for action in self.legal_actions:
            for chance_code in range(16):
                try:
                    child_N = self.children[(action, chance_code)].N
                    U = self.config.UCT_exploration_constant * self.policy[action] * np.sqrt(self.N) / (child_N + 1)
                    Q = self.children[(action, chance_code)].get_value()
                except KeyError:
                    U = self.config.UCT_exploration_constant * self.policy[action] * np.sqrt(self.N)
                    Q = 0
                UCT_scores[(action, chance_code)] = Q + U

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


def get_temperature(step_number, training=True, config=MCTSConfig()):
    if not training:
        return 0.  # greedy en évaluation
    else:
        return config.base_temperature
    # if step_number < 10:
    #     return config.base_temperature
    # else:
    #     return 0.1   # quasi-greedy en fin de partie


def run_mcts(agent, game, step_number, config=MCTSConfig(), root=None, training=True):
    """
    perform multiple MCTS simulations, returning root node value and MCTS-based policy
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
                action, chance_code = random_max_in_dict(UCT_scores)
                node = node.get_or_create_child(action, chance_code)
                if node.N == 0:
                    is_leaf = True
            else:
                is_leaf = True
        backpropagate(node)

    value = root.get_value()

    # Calcul de la policy avec température
    visit_counts = np.zeros((22,))
    for k, v in root.children.items():
        index = k[0]
        visit_counts[index] += v.N

    temperature = get_temperature(step_number, training=training, config=config)

    if temperature < 0.01:  # cas greedy, évite division par zéro
        policy = np.zeros((22,))
        policy[np.argmax(visit_counts)] = 1.0
    else:
        counts_temp = visit_counts ** (1 / temperature)
        policy = counts_temp / counts_temp.sum()

    new_root = root
    new_root.parent = None

    return value, policy, new_root
