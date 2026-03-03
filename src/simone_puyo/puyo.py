import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
import matplotlib.colors


# negative reward when action results in game over
GAMEOVER_REWARD = -np.sqrt(100 + 1) - 1

# reward function for chains 1 through 19
reward_dict = {
}
for i in range(20):
    reward_dict[int(i)] = np.sqrt(i**2.5 + 1) - 1

# color parameters for plotting game boards
cvals = [0, 1, 2, 3, 4, 5]
colors = ["c", "r", "g", "b", "m", "white"]
norm = plt.Normalize(min(cvals), max(cvals))
color_tuples = list(zip(map(norm, cvals), colors))
puyo_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", color_tuples)

# dictionary map for chance events: random new puyo pair (tsumo)
dict_chance_tsumos = {
    0: [1, 1],
    1: [1, 2],
    2: [1, 3],
    3: [1, 4],
    4: [2, 1],
    5: [2, 2],
    6: [2, 3],
    7: [2, 4],
    8: [3, 1],
    9: [3, 2],
    10: [3, 3],
    11: [3, 4],
    12: [4, 1],
    13: [4, 2],
    14: [4, 3],
    15: [4, 4]
}


def get_chance_code(tsumo):
    """
    Reverse map to obtain chance code from random new puyo pair
    """
    if tsumo == [1, 1]:
        return 0
    if tsumo == [1, 2]:
        return 1
    if tsumo == [1, 3]:
        return 2
    if tsumo == [1, 4]:
        return 3
    if tsumo == [2, 1]:
        return 4
    if tsumo == [2, 2]:
        return 5
    if tsumo == [2, 3]:
        return 6
    if tsumo == [2, 4]:
        return 7
    if tsumo == [3, 1]:
        return 8
    if tsumo == [3, 2]:
        return 9
    if tsumo == [3, 3]:
        return 10
    if tsumo == [3, 4]:
        return 11
    if tsumo == [4, 1]:
        return 12
    if tsumo == [4, 2]:
        return 13
    if tsumo == [4, 3]:
        return 14
    if tsumo == [4, 4]:
        return 15
    else:
        raise ValueError('Invalid tsumo.')


def array_num2onehot(array):
    """
    transform numeric representation of an array to its one-hot encoding
    """
    nrow = array.shape[0]
    ncol = array.shape[1]
    onehot_array = np.zeros((nrow, ncol, 4), dtype=np.float32)

    for color in range(1, 5):  # Colors 1-4
        onehot_array[:, :, color - 1] = (array == color)

    return onehot_array


# def find_placing_index_vectorized(board):
#     """
#     Find the lowest available space in all board columns at once.
#     Returns array of indices of shape (6,)
#     """
#     # For each column, find first non-zero from bottom
#     # If column is empty, return nrow (13)
#     # If column is full, return -1
#     nrow = board.shape[0]
#     indices = np.zeros(6, dtype=np.int32)
#
#     for col in range(6):
#         column = board[:, col]
#         # Find first non-zero element
#         nonzero_idx = np.where(column != 0)[0]
#         if len(nonzero_idx) == 0:
#             # Column is empty
#             indices[col] = nrow - 1
#         else:
#             # Place just above first non-zero
#             indices[col] = nonzero_idx[0] - 1
#
#     return indices


def get_legal_actions(board):
    """
    Return a list of all legal moves on the current state of the board.
    """
    # Start with all moves legal
    legal_actions = list(range(22))

    # Check top row for blocked columns
    top_row = board[1, :]
    blocked_cols = np.where(top_row != 0)[0]

    illegal_actions = []

    # Vertical moves (0-11) are illegal if column is full
    for col in blocked_cols:
        illegal_actions.extend([col, col + 6])

    # Horizontal moves (12-21) are illegal if one column is full
    for move in range(12, 17):
        col1 = move - 12
        col2 = col1 + 1
        if board[1, col1] != 0 or board[1, col2] != 0:
            illegal_actions.extend([move, move + 5])

    legal_actions = [move for move in legal_actions if move not in illegal_actions]

    return legal_actions


class TsumoQueue:
    """
    Class for queue of upcoming puyo pairs
    """
    def __init__(self):
        self.queue = np.zeros((3, 2), dtype=np.int32)
        self.onehot_queue = None
        self.current = np.zeros((1, 2), dtype=np.int32)
        self.next1 = np.zeros((1, 2), dtype=np.int32)
        self.next2 = np.zeros((1, 2), dtype=np.int32)

    def update_pairs(self):
        """
        update individual pairs in the queue from the global queue
        """
        self.current[0, :] = self.queue[0, :]
        self.next1[0, :] = self.queue[1, :]
        self.next2[0, :] = self.queue[2, :]

    def start_queue(self):
        """
        initialize the queue, with the first two pairs drawn from 3 colors and the third from 4 colors
        """
        self.queue[0, :] = np.random.randint(3, size=2) + 1
        self.queue[1, :] = np.random.randint(3, size=2) + 1
        self.queue[2, :] = np.random.randint(4, size=2) + 1
        self.update_pairs()

    def progress_queue(self):
        """
        progress the queue by moving the next pieces by one step towards the current pair and generate a new last pair
        """
        self.queue[0, :] = self.queue[1, :]
        self.queue[1, :] = self.queue[2, :]
        self.queue[2, :] = np.random.randint(4, size=2) + 1
        self.update_pairs()

    def insert_last_in_queue(self, code):
        """
        Insert new puyo pair in the queue's last spot from chance code
        """
        self.queue[2, :] = np.array(dict_chance_tsumos[code], dtype=np.int32)
        self.update_pairs()

    def update_onehot_queue(self):
        self.onehot_queue = array_num2onehot(self.queue)

    def get_num_queue(self):
        """
        return the numeric representation of the global queue
        """
        return self.queue

    def get_onehot_queue(self):
        """
        return the one-hot representation of the global queue
        """
        return self.onehot_queue


class Board:
    """
    Class for the game board
    """
    def __init__(self):
        self.num_board = np.zeros((13, 6), dtype=np.int32)
        self.onehot_board = np.zeros((4, 13, 6), dtype=np.float32)
        self.nrow = 13
        self.ncol = 6

        self.placing_indices = np.ones((6, ), dtype=np.int32) * (self.nrow - 1)

    def place_tsumo_num(self, num_tsumo, move):
        """
        Place numeric representation of puyo on the board.
        """
        puyo1, puyo2 = num_tsumo[0, 0], num_tsumo[0, 1]

        # Precompute placing indices for all columns (vectorized)
        # placing_indices = find_placing_index_vectorized(self.num_board)

        if move < 6:
            # Vertical moves (0-5): puyo1 bottom, puyo2 top
            col_idx = move
            idx_puyo1 = self.placing_indices[col_idx]
            idx_puyo2 = idx_puyo1 - 1
            if idx_puyo1 >= 0:
                self.num_board[idx_puyo1, col_idx] = puyo1
            if idx_puyo2 >= 0:
                self.num_board[idx_puyo2, col_idx] = puyo2
            self.placing_indices[col_idx] -= 2

        elif move < 12:
            # Vertical moves (6-11): puyo2 bottom, puyo1 top
            col_idx = move - 6
            idx_puyo1 = self.placing_indices[col_idx]
            idx_puyo2 = idx_puyo1 - 1
            if idx_puyo1 >= 0:
                self.num_board[idx_puyo1, col_idx] = puyo2
            if idx_puyo2 >= 0:
                self.num_board[idx_puyo2, col_idx] = puyo1
            self.placing_indices[col_idx] -= 2

        elif move < 17:
            # Horizontal moves (12-16): puyo1 left, puyo2 right
            col1_idx = move - 12
            col2_idx = col1_idx + 1
            idx_puyo1 = self.placing_indices[col1_idx]
            idx_puyo2 = self.placing_indices[col2_idx]
            if idx_puyo1 >= 0:
                self.num_board[idx_puyo1, col1_idx] = puyo1
            if idx_puyo2 >= 0:
                self.num_board[idx_puyo2, col2_idx] = puyo2
            self.placing_indices[col1_idx] -= 1
            self.placing_indices[col2_idx] -= 1

        else:
            # Horizontal moves (17-21): puyo2 left, puyo1 right
            col1_idx = move - 17
            col2_idx = col1_idx + 1
            idx_puyo1 = self.placing_indices[col1_idx]
            idx_puyo2 = self.placing_indices[col2_idx]
            if idx_puyo1 >= 0:
                self.num_board[idx_puyo1, col1_idx] = puyo2
            if idx_puyo2 >= 0:
                self.num_board[idx_puyo2, col2_idx] = puyo1
            self.placing_indices[col1_idx] -= 1
            self.placing_indices[col2_idx] -= 1

    def update_onehot_board(self):
        """
        Update the one-hot representation of the game board
        """
        self.onehot_board = array_num2onehot(self.num_board)

    def gravity(self):
        """
        Apply gravity to the current state of the board.
        """
        # For each column, move all non-zero elements to the bottom
        for col_idx in range(self.ncol):
            column = self.num_board[:, col_idx]
            # Get non-zero elements
            nonzero_elements = column[column != 0]
            # Create new column with zeros at top, non-zeros at bottom
            n_nonzero = len(nonzero_elements)
            n_zeros = self.nrow - n_nonzero
            new_column = np.concatenate([np.zeros(n_zeros, dtype=np.int32), nonzero_elements])
            self.num_board[:, col_idx] = new_column

    def chain_step(self):
        """
        Resolve one step of the chain on the current board.
        """
        has_chain = False

        # Work on subboard (excluding top row for game over cell)
        subboard = self.num_board[1:, :]

        # Process all 4 colors in parallel
        all_remove_indices = []

        for color in range(1, 5):
            # Create binary mask for this color
            color_mask = (subboard == color).astype(np.int32)

            # Label connected components
            labeled_arr, n_components = label(color_mask)

            # Find components with 4+ puyos
            for comp_id in range(1, n_components + 1):
                component_mask = (labeled_arr == comp_id)
                group_size = np.sum(component_mask)

                if group_size >= 4:
                    has_chain = True
                    # Get indices of this component
                    indices = np.argwhere(component_mask)
                    all_remove_indices.extend(indices.tolist())

        # Remove all marked puyos at once (vectorized)
        if all_remove_indices:
            for idx_pair in all_remove_indices:
                # +1 because we're working on subboard (row 1+)
                self.num_board[idx_pair[0] + 1, idx_pair[1]] = 0

        # Apply gravity
        if has_chain:
            self.gravity()

        return has_chain

    def resolve_chain(self):
        """
        resolve the full chain on the current board
        """
        chain_length = 0
        has_chain = True
        while has_chain:
            has_chain = self.chain_step()
            if has_chain:
                chain_length += 1

        return chain_length

    def check_gameover(self):
        """
        check if the gameover cell is filled or not
        return a boolean which is True if game is over, False otherwise
        """
        gameover = False
        if self.num_board[1, 2] != 0:
            gameover = True
        return gameover

    def get_num_board(self):
        """
        return the numeric representation of the board
        """
        return self.num_board

    def get_onehot_board(self):
        """
        return the one-hot reprensation of the board
        """
        return self.onehot_board


class GameState:
    """
    Wrapper class for the whole game state: puyo pair queue and game board
    """
    def __init__(self, board, queue):
        self.board = board
        self.queue = queue

    def make_onehot_state(self):
        """
        prepares a combined one-hot representation of both the game board and the puyo queue
        """
        self.board.update_onehot_board()
        self.queue.update_onehot_queue()
        onehot_state = np.zeros((self.board.num_board.shape[0] + 1,
                                 self.board.num_board.shape[1], 4), dtype=np.float32)
        onehot_state[:-1, :, :] = self.board.onehot_board
        onehot_state[-1, :2, :] = self.queue.onehot_queue[0, :, :]
        onehot_state[-1, 2:4, :] = self.queue.onehot_queue[1, :, :]
        onehot_state[-1, 4:6, :] = self.queue.onehot_queue[2, :, :]

        return onehot_state


class PuyoGame:
    """
    Class for a complete game of single-player, turn-based Puyo Puyo
    """
    def __init__(self, max_moves):
        self.max_moves = max_moves

        queue = TsumoQueue()
        queue.start_queue()
        board = Board()
        self.state = GameState(board, queue)
        self.n_step = 0

    def reset(self):
        """
        resets the whole game
        """
        self.__init__(self.max_moves)
        return self.get_input()

    def copy(self):
        """
        makes a copy of itself, used in MCTS
        """
        new_game = PuyoGame(self.max_moves)
        new_game.n_step = self.n_step
        new_board = Board()
        new_board.num_board[:, :] = self.state.board.num_board[:, :]
        new_queue = TsumoQueue()
        new_queue.queue[:, :] = self.state.queue.queue[:, :]
        new_queue.update_pairs()
        new_game.state = GameState(new_board, new_queue)

        return new_game

    def get_input(self):
        """
        returns the one-hot representation of the game state
        """
        return self.state.make_onehot_state()

    def get_legal_actions(self):
        """
        returns current legal actions
        """
        return get_legal_actions(self.state.board.num_board)

    def step(self, action):
        """
        place the current pair of the queue on the board according to the chosen move
        progress the queue, resolve the chain if applicable, check for a gameover
        return chain length (int) and gameover boolean (True if gameover, False otherwise)
        """
        self.state.board.place_tsumo_num(self.state.queue.current, action)
        self.state.queue.progress_queue()
        chain_length = self.state.board.resolve_chain()
        gameover = self.state.board.check_gameover()
        self.n_step += 1

        reward = GAMEOVER_REWARD if gameover else reward_dict[chain_length]

        done = True if self.n_step >= self.max_moves or gameover else False

        return self.get_input(), reward, done

    def display_screen(self, savepath=None, title='screen_frame'):
        """
        displays a numeric representation of a board on a pyplot figure
        figure is saved as png if a file name is provided (e.g. filename = './figures/myfig.png')
        """
        fig, ax = plt.subplots()

        screen = np.zeros((13, 8))
        screen[:, :6] = self.state.board.num_board
        screen[:, 6] = 5.
        screen[0, 7] = self.state.queue.queue[0, 0]
        screen[1, 7] = self.state.queue.queue[0, 1]
        screen[2, 7] = 5.
        screen[3, 7] = self.state.queue.queue[1, 0]
        screen[4, 7] = self.state.queue.queue[1, 1]
        screen[5, 7] = 5.
        screen[6, 7] = self.state.queue.queue[2, 0]
        screen[7, 7] = self.state.queue.queue[2, 1]
        screen[8:, 7] = 5.

        ax.imshow(screen, cmap=puyo_cmap)
        ax.set_title(title)
        plt.axis('off')
        if savepath is not None:
            plt.savefig(savepath)
            plt.close(fig)
        else:
            plt.show()