import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.colors


# negative reward when action results in game over
GAMEOVER_REWARD = -(np.sqrt(100 + 1) - 1)

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


# def array_num2onehot(array):
#     """
#     transform numeric representation of an array to its one-hot encoding
#     """
#     nrow = array.shape[0]
#     ncol = array.shape[1]
#     onehot_array = np.zeros((nrow, ncol, 4), dtype=np.float32)
#
#     for color in range(1, 5):  # Colors 1-4
#         onehot_array[:, :, color - 1] = (array == color)
#
#     return onehot_array


@njit(cache=True, nogil=True)
def array_num2onehot(array):
    """
    transform numeric representation of an array to its one-hot encoding
    """
    nrow, ncol = array.shape
    onehot_array = np.zeros((nrow, ncol, 4), dtype=np.float32)
    for r in range(nrow):
        for c in range(ncol):
            v = array[r, c]
            if 1 <= v <= 4:
                onehot_array[r, c, v - 1] = 1.0
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


# def find_placing_index_vectorized(board):
#     """
#     Find the lowest available space in all board columns at once.
#     Returns array of indices of shape (6,)
#     """
#     nrow = board.shape[0]
#     nonzero = (board != 0)
#     has_any = nonzero.any(axis=0)
#     # argmax sur un booleen renvoie l'indice du premier True ; repli sur nrow
#     # si la colonne est entierement vide (aucun True trouve).
#     first_nonzero = np.where(has_any, nonzero.argmax(axis=0), nrow)
#     return (first_nonzero - 1).astype(np.int32)


@njit(cache=True)
def find_placing_index_vectorized(board):
    """
    Find the lowest available space in all board columns at once.
    Returns array of indices of shape (6,)
    """
    nrow, ncol = board.shape
    indices = np.empty(ncol, dtype=np.int32)
    for c in range(ncol):
        idx = nrow - 1
        for r in range(nrow):
            if board[r, c] != 0:
                idx = r - 1
                break
        indices[c] = idx
    return indices


@njit(cache=True, nogil=True)
def _find_chain_removals(subboard):
    """
    Detecte, pour TOUTES les couleurs en une seule passe (chaque cellule
    n'appartient qu'a une couleur), les composantes connexes (4-connexite)
    de taille >= 4. Remplace scipy.ndimage.label, specialise pour ce plateau.
    Retourne (has_chain, remove_mask).
    """
    nrow, ncol = subboard.shape
    visited = np.zeros((nrow, ncol), dtype=np.bool_)
    remove_mask = np.zeros((nrow, ncol), dtype=np.bool_)
    has_chain = False

    stack_r = np.empty(nrow * ncol, dtype=np.int32)
    stack_c = np.empty(nrow * ncol, dtype=np.int32)
    comp_r = np.empty(nrow * ncol, dtype=np.int32)
    comp_c = np.empty(nrow * ncol, dtype=np.int32)

    for r0 in range(nrow):
        for c0 in range(ncol):
            color = subboard[r0, c0]
            if color == 0 or visited[r0, c0]:
                continue

            size = 0
            sp = 0
            stack_r[sp] = r0
            stack_c[sp] = c0
            sp += 1
            visited[r0, c0] = True

            while sp > 0:
                sp -= 1
                r = stack_r[sp]
                c = stack_c[sp]
                comp_r[size] = r
                comp_c[size] = c
                size += 1

                if r + 1 < nrow and not visited[r + 1, c] and subboard[r + 1, c] == color:
                    visited[r + 1, c] = True; stack_r[sp] = r + 1; stack_c[sp] = c; sp += 1
                if r - 1 >= 0 and not visited[r - 1, c] and subboard[r - 1, c] == color:
                    visited[r - 1, c] = True; stack_r[sp] = r - 1; stack_c[sp] = c; sp += 1
                if c + 1 < ncol and not visited[r, c + 1] and subboard[r, c + 1] == color:
                    visited[r, c + 1] = True; stack_r[sp] = r; stack_c[sp] = c + 1; sp += 1
                if c - 1 >= 0 and not visited[r, c - 1] and subboard[r, c - 1] == color:
                    visited[r, c - 1] = True; stack_r[sp] = r; stack_c[sp] = c - 1; sp += 1

            if size >= 4:
                has_chain = True
                for i in range(size):
                    remove_mask[comp_r[i], comp_c[i]] = True

    return has_chain, remove_mask


@njit(cache=True, nogil=True)
def _apply_gravity_numba(board):
    """
    Compacte chaque colonne vers le bas en preservant l'ordre relatif des
    elements non nuls. Remplace la version numpy (concatenate par colonne).
    Modifie `board` en place et le retourne.
    """
    nrow, ncol = board.shape
    for c in range(ncol):
        write_idx = nrow - 1
        for r in range(nrow - 1, -1, -1):
            if board[r, c] != 0:
                if write_idx != r:
                    board[write_idx, c] = board[r, c]
                    board[r, c] = 0
                write_idx -= 1
    return board


def _resolve_chain_on_copy(board):
    """
    Resout la chaine sur `board` (deja modifie en place, typiquement une
    copie jetable) exactement comme Board.resolve_chain()/chain_step() :
    detection sur les 12 lignes visibles (subboard), gravite sur les 13
    lignes y compris la ligne 0 cachee. Retourne la longueur de chaine.

    Reste en Python pur (comme chain_step()/resolve_chain()) : numba ne
    supporte pas l'assignation par masque booleen 2D (subboard[mask]=0)
    en mode nopython. Seuls les noyaux de calcul appeles ici
    (_find_chain_removals, _apply_gravity_numba) sont numba-jit.
    """
    chain_length = 0
    has_chain = True
    while has_chain:
        subboard = board[1:, :]
        has_chain, remove_mask = _find_chain_removals(subboard)
        if has_chain:
            subboard[remove_mask] = 0
            _apply_gravity_numba(board)
            chain_length += 1
    return chain_length


def chain_potential(board):
    """
    Potentiel de chaine latent d'un plateau : pour chacune des 6 colonnes
    et chacune des 4 couleurs (24 hypotheses), teste l'ajout d'UN puyo de
    cette couleur au sommet de la colonne -- sur une copie, `board` n'est
    jamais modifie -- et resout la chaine qui en resulterait. Retourne la
    longueur de chaine maximale obtenue sur les 24 hypotheses (0 si
    aucune ne declenche de chaine). Utilise pour un reward shaping
    optionnel (voir RewardConfig dans actor.py) et/ou comme diagnostic
    independant : ne participe jamais a la vraie regle du jeu.
    """
    nrow, ncol = board.shape
    best = 0
    placing_indices = find_placing_index_vectorized(board)

    for c in range(ncol):
        idx = placing_indices[c]
        if idx < 0:
            continue  # colonne pleine, pas de place pour un puyo de plus
        for color in range(1, 5):
            trial = board.copy()
            trial[idx, c] = color
            chain_length = _resolve_chain_on_copy(trial)
            if chain_length > best:
                best = chain_length

    return best


@njit(cache=True, nogil=True)
def _legal_actions_mask_numba(top_row):
    """
    Retourne un masque booleen de taille 22 (legal=True) a partir de la
    ligne du haut jouable (6 colonnes).
    """
    mask = np.ones(22, dtype=np.bool_)
    for col in range(6):
        if top_row[col] != 0:
            mask[col] = False
            mask[col + 6] = False
    for move in range(12, 17):
        col1 = move - 12
        col2 = col1 + 1
        if top_row[col1] != 0 or top_row[col2] != 0:
            mask[move] = False
            mask[move + 5] = False
    return mask


@njit(cache=True, nogil=True)
def _place_tsumo_numba(board, puyo1, puyo2, move):
    placing_indices = find_placing_index_vectorized(board)

    if move < 6:
        col_idx = move
        idx_puyo1 = placing_indices[col_idx]
        idx_puyo2 = idx_puyo1 - 1
        if idx_puyo1 >= 0:
            board[idx_puyo1, col_idx] = puyo1
        if idx_puyo2 >= 0:
            board[idx_puyo2, col_idx] = puyo2

    elif move < 12:
        col_idx = move - 6
        idx_puyo1 = placing_indices[col_idx]
        idx_puyo2 = idx_puyo1 - 1
        if idx_puyo1 >= 0:
            board[idx_puyo1, col_idx] = puyo2
        if idx_puyo2 >= 0:
            board[idx_puyo2, col_idx] = puyo1

    elif move < 17:
        col1_idx = move - 12
        col2_idx = col1_idx + 1
        idx_puyo1 = placing_indices[col1_idx]
        idx_puyo2 = placing_indices[col2_idx]
        if idx_puyo1 >= 0:
            board[idx_puyo1, col1_idx] = puyo1
        if idx_puyo2 >= 0:
            board[idx_puyo2, col2_idx] = puyo2

    else:
        col1_idx = move - 17
        col2_idx = col1_idx + 1
        idx_puyo1 = placing_indices[col1_idx]
        idx_puyo2 = placing_indices[col2_idx]
        if idx_puyo1 >= 0:
            board[idx_puyo1, col1_idx] = puyo2
        if idx_puyo2 >= 0:
            board[idx_puyo2, col2_idx] = puyo1


@njit(cache=True, nogil=True)
def _compute_group_size_map(subboard):
    """
    Pour chaque cellule occupee du plateau jouable (12x6), retourne la taille
    de sa composante connexe normalisee par 3. Toujours dans [0, 1] puisque
    les groupes >= 4 sont resolus par resolve_chain avant tout appel a get_input.
    """
    nrow, ncol = subboard.shape
    visited = np.zeros((nrow, ncol), dtype=np.bool_)
    size_map = np.zeros((nrow, ncol), dtype=np.float32)
    stack_r = np.empty(nrow * ncol, dtype=np.int32)
    stack_c = np.empty(nrow * ncol, dtype=np.int32)
    comp_r  = np.empty(nrow * ncol, dtype=np.int32)
    comp_c  = np.empty(nrow * ncol, dtype=np.int32)

    for r0 in range(nrow):
        for c0 in range(ncol):
            color = subboard[r0, c0]
            if color == 0 or visited[r0, c0]:
                continue
            size = 0; sp = 0
            stack_r[sp] = r0; stack_c[sp] = c0; sp += 1
            visited[r0, c0] = True
            while sp > 0:
                sp -= 1
                r = stack_r[sp]; c = stack_c[sp]
                comp_r[size] = r; comp_c[size] = c; size += 1
                if r+1 < nrow and not visited[r+1,c] and subboard[r+1,c]==color:
                    visited[r+1,c]=True; stack_r[sp]=r+1; stack_c[sp]=c; sp+=1
                if r-1 >= 0  and not visited[r-1,c] and subboard[r-1,c]==color:
                    visited[r-1,c]=True; stack_r[sp]=r-1; stack_c[sp]=c; sp+=1
                if c+1 < ncol and not visited[r,c+1] and subboard[r,c+1]==color:
                    visited[r,c+1]=True; stack_r[sp]=r; stack_c[sp]=c+1; sp+=1
                if c-1 >= 0  and not visited[r,c-1] and subboard[r,c-1]==color:
                    visited[r,c-1]=True; stack_r[sp]=r; stack_c[sp]=c-1; sp+=1
            normalized = size / 3.0
            for i in range(size):
                size_map[comp_r[i], comp_c[i]] = normalized

    return size_map


# def get_legal_actions(board):
#     """
#     Return a list of all legal moves on the current state of the board.
#     """
#     # Start with all moves legal
#     legal_actions = list(range(22))
#
#     # Check top row for blocked columns
#     top_row = board[1, :]
#     blocked_cols = np.where(top_row != 0)[0]
#
#     illegal_actions = []
#
#     # Vertical moves (0-11) are illegal if column is full
#     for col in blocked_cols:
#         illegal_actions.extend([col, col + 6])
#
#     # Horizontal moves (12-21) are illegal if one column is full
#     for move in range(12, 17):
#         col1 = move - 12
#         col2 = col1 + 1
#         if board[1, col1] != 0 or board[1, col2] != 0:
#             illegal_actions.extend([move, move + 5])
#
#     legal_actions = [move for move in legal_actions if move not in illegal_actions]
#
#     return legal_actions


def get_legal_actions(board):
    """
    Return a list of all legal moves on the current state of the board.
    """
    mask = _legal_actions_mask_numba(board[1, :])
    return np.flatnonzero(mask).tolist()


def _build_action_mirror_map():
    """
    Table de correspondance action <-> action miroir (symetrie gauche-droite,
    colonne c -> 5-c), deduite de _place_tsumo_numba :
      0-5   : vertical, puyo1 bas/puyo2 haut, colonne = action
      6-11  : vertical, puyo2 bas/puyo1 haut, colonne = action-6
      12-16 : horizontal, puyo1 gauche/puyo2 droite
      17-21 : horizontal, puyo2 gauche/puyo1 droite
    """
    mirror_map = np.empty(22, dtype=np.int64)
    for a in range(6):
        mirror_map[a] = 5 - a
    for a in range(6, 12):
        mirror_map[a] = 17 - a
    for a in range(12, 22):
        mirror_map[a] = 33 - a
    return mirror_map


ACTION_MIRROR_MAP = _build_action_mirror_map()


# Position de la case de game over (row=1, col=2 dans num_board / onehot_state)
# et de sa colonne symetrique (row=1, col=3). check_gameover() ne teste QUE la
# colonne 2 : le moteur de jeu n'est donc pas invariant par symetrie horizontale.
GAMEOVER_ROW = 1
GAMEOVER_COL = 2
GAMEOVER_COL_MIRROR = 5 - GAMEOVER_COL  # = 3


def can_mirror_observation(observation):
    """
    False si la case de game over (row=1, col=2) OU sa symetrique (row=1, col=3)
    est occupee dans cet etat.

    - col=2 occupee : ne devrait en theorie jamais arriver pour une observation
      stockee (check_gameover() ne teste que cette colonne, et l'episode
      s'arrete des que done=True), mais on garde le test par securite.
    - col=3 occupee : le miroir deplacerait ce puyo en colonne 2, produisant un
      etat qui ressemble a un game over (colonne 2 remplie) alors qu'il serait
      associe a une politique/valeur de continuation normale -- un etat hors
      distribution que le vrai moteur ne produit jamais pour une transition
      non terminale.
    """
    col2_occupied = np.any(observation[GAMEOVER_ROW, GAMEOVER_COL, :4])
    col3_occupied = np.any(observation[GAMEOVER_ROW, GAMEOVER_COL_MIRROR, :4])
    return not col2_occupied and not col3_occupied


def mirror_observation(observation):
    """
    Symetrie horizontale d'un etat one-hot (13, 6, 30) : inversion de l'axe
    des colonnes. Valide pour tous les canaux : couleurs et taille de groupe
    (spatiaux, correctement reflechis), hauteur de colonne (spatial, colonnes
    permutees comme attendu), canaux de queue (constants sur les colonnes,
    donc inchanges par l'inversion).
    """
    return observation[:, ::-1, :].copy()


def mirror_policy(policy):
    """
    Symetrie horizontale d'un vecteur de politique (22,).
    """
    return policy[ACTION_MIRROR_MAP]


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

    # def place_tsumo_num(self, num_tsumo, move):
    #     """
    #     Place numeric representation of puyo on the board.
    #     """
    #     puyo1, puyo2 = num_tsumo[0, 0], num_tsumo[0, 1]
    #
    #     # Precompute placing indices for all columns (vectorized)
    #     placing_indices = find_placing_index_vectorized(self.num_board)
    #
    #     if move < 6:
    #         # Vertical moves (0-5): puyo1 bottom, puyo2 top
    #         col_idx = move
    #         idx_puyo1 = placing_indices[col_idx]
    #         idx_puyo2 = idx_puyo1 - 1
    #         if idx_puyo1 >= 0:
    #             self.num_board[idx_puyo1, col_idx] = puyo1
    #         if idx_puyo2 >= 0:
    #             self.num_board[idx_puyo2, col_idx] = puyo2
    #
    #     elif move < 12:
    #         # Vertical moves (6-11): puyo2 bottom, puyo1 top
    #         col_idx = move - 6
    #         idx_puyo1 = placing_indices[col_idx]
    #         idx_puyo2 = idx_puyo1 - 1
    #         if idx_puyo1 >= 0:
    #             self.num_board[idx_puyo1, col_idx] = puyo2
    #         if idx_puyo2 >= 0:
    #             self.num_board[idx_puyo2, col_idx] = puyo1
    #
    #     elif move < 17:
    #         # Horizontal moves (12-16): puyo1 left, puyo2 right
    #         col1_idx = move - 12
    #         col2_idx = col1_idx + 1
    #         idx_puyo1 = placing_indices[col1_idx]
    #         idx_puyo2 = placing_indices[col2_idx]
    #         if idx_puyo1 >= 0:
    #             self.num_board[idx_puyo1, col1_idx] = puyo1
    #         if idx_puyo2 >= 0:
    #             self.num_board[idx_puyo2, col2_idx] = puyo2
    #
    #     else:
    #         # Horizontal moves (17-21): puyo2 left, puyo1 right
    #         col1_idx = move - 17
    #         col2_idx = col1_idx + 1
    #         idx_puyo1 = placing_indices[col1_idx]
    #         idx_puyo2 = placing_indices[col2_idx]
    #         if idx_puyo1 >= 0:
    #             self.num_board[idx_puyo1, col1_idx] = puyo2
    #         if idx_puyo2 >= 0:
    #             self.num_board[idx_puyo2, col2_idx] = puyo1

    def place_tsumo_num(self, num_tsumo, move):
        """
        Place numeric representation of puyo on the board.
        """
        puyo1, puyo2 = int(num_tsumo[0, 0]), int(num_tsumo[0, 1])
        _place_tsumo_numba(self.num_board, puyo1, puyo2, move)

    def update_onehot_board(self):
        """
        Update the one-hot representation of the game board
        """
        self.onehot_board = array_num2onehot(self.num_board)

    # def gravity(self):
    #     """
    #     Apply gravity to the current state of the board.
    #     """
    #     # For each column, move all non-zero elements to the bottom
    #     for col_idx in range(self.ncol):
    #         column = self.num_board[:, col_idx]
    #         # Get non-zero elements
    #         nonzero_elements = column[column != 0]
    #         # Create new column with zeros at top, non-zeros at bottom
    #         n_nonzero = len(nonzero_elements)
    #         n_zeros = self.nrow - n_nonzero
    #         new_column = np.concatenate([np.zeros(n_zeros, dtype=np.int32), nonzero_elements])
    #         self.num_board[:, col_idx] = new_column

    def gravity(self):
        """
        Apply gravity to the current state of the board.
        """
        _apply_gravity_numba(self.num_board)

    # def chain_step(self):
    #     """
    #     Resolve one step of the chain on the current board.
    #     """
    #     has_chain = False
    #
    #     # Work on subboard (excluding top row for game over cell)
    #     subboard = self.num_board[1:, :]
    #
    #     # Process all 4 colors in parallel
    #     all_remove_indices = []
    #
    #     for color in range(1, 5):
    #         # Create binary mask for this color
    #         color_mask = (subboard == color).astype(np.int32)
    #
    #         # Label connected components
    #         labeled_arr, n_components = label(color_mask)
    #
    #         # Find components with 4+ puyos
    #         for comp_id in range(1, n_components + 1):
    #             component_mask = (labeled_arr == comp_id)
    #             group_size = np.sum(component_mask)
    #
    #             if group_size >= 4:
    #                 has_chain = True
    #                 # Get indices of this component
    #                 indices = np.argwhere(component_mask)
    #                 all_remove_indices.extend(indices.tolist())
    #
    #     # Remove all marked puyos at once (vectorized)
    #     if all_remove_indices:
    #         for idx_pair in all_remove_indices:
    #             # +1 because we're working on subboard (row 1+)
    #             self.num_board[idx_pair[0] + 1, idx_pair[1]] = 0
    #
    #     # Apply gravity
    #     if has_chain:
    #         self.gravity()
    #
    #     return has_chain

    def chain_step(self):
        """
        Resolve one step of the chain on the current board.
        """
        subboard = self.num_board[1:, :]  # vue, pas une copie
        has_chain, remove_mask = _find_chain_removals(subboard)

        if has_chain:
            subboard[remove_mask] = 0  # ecrit directement dans self.num_board via la vue
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
        Representation one-hot combinee du plateau et de la queue.
        Shape : (13, 6, 30)

        Channels:
          0-3  : board one-hot couleurs (4 canaux, lignes 0-12)
          4    : taille de composante connexe normalisee (lignes 1-12, 0 ailleurs)
          5    : hauteur normalisee de chaque colonne, broadcast sur toutes les lignes
          6-29 : queue one-hot, 3 paires x 2 puyos x 4 couleurs = 24 canaux binaires
                 chaque canal est constant (broadcast) sur toutes les positions (r, c)
        """
        self.board.update_onehot_board()
        board = self.board.num_board  # (13, 6) int32

        onehot_state = np.zeros((13, 6, 30), dtype=np.float32)

        # Channels 0-3 : board one-hot
        onehot_state[:, :, :4] = self.board.onehot_board

        # Channel 4 : taille de composante connexe (uniquement lignes jouables 1-12)
        onehot_state[1:, :, 4] = _compute_group_size_map(board[1:, :])

        # Channel 5 : hauteur de colonne normalisee (meme valeur sur toute la hauteur)
        heights = np.count_nonzero(board[1:, :], axis=0) / 12.0  # (6,)
        onehot_state[:, :, 5] = heights  # broadcast sur les 13 lignes

        # Channels 6-29 : queue one-hot
        # Ordre : pair0_puyo0, pair0_puyo1, pair1_puyo0, pair1_puyo1, pair2_puyo0, pair2_puyo1
        # Chaque puyo occupe 4 channels (one-hot sur les 4 couleurs), broadcast spatial
        ch = 6
        for pair_idx in range(3):
            for puyo_idx in range(2):
                color = int(self.queue.queue[pair_idx, puyo_idx])  # 1-4
                if 1 <= color <= 4:
                    onehot_state[:, :, ch + color - 1] = 1.0
                ch += 4

        return onehot_state


class PuyoGame:
    """
    Class for a complete game of single-player, turn-based Puyo Puyo
    """
    def __init__(self, max_moves, copy=False):
        self.max_moves = max_moves

        if not copy:
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
        new_game = PuyoGame(self.max_moves, copy=True)
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

        `done` is True whenever the episode stops (real game over OR max_moves
        reached). `gameover` specifically flags a REAL game over, as opposed to
        a truncation by max_moves. The distinction matters downstream: a
        truncated episode should be bootstrapped with an estimated value
        instead of being treated as if no future reward were possible.
        """
        self.state.board.place_tsumo_num(self.state.queue.current, action)
        self.state.queue.progress_queue()
        chain_length = self.state.board.resolve_chain()
        gameover = self.state.board.check_gameover()
        self.n_step += 1

        reward = GAMEOVER_REWARD if gameover else reward_dict[chain_length]

        done = True if self.n_step >= self.max_moves or gameover else False

        return self.get_input(), reward, done, gameover

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