import numpy as np


def random_max_in_dict(d):
    """
    chooses a random key (action and chance code) from those with maximum value
    """
    list_max_action = []
    list_max_chance_code = []
    max_v = -np.inf

    for k, v in d.items():
        if v > max_v:
            list_max_action = [k[0]]
            list_max_chance_code = [k[1]]
        elif v == max_v:
            list_max_action.append(k[0])
            list_max_chance_code.append(k[1])

    indices = range(len(list_max_action))
    random_index = np.random.choice(indices)

    action = list_max_action[random_index]
    chance_code = list_max_chance_code[random_index]

    return action, chance_code


def random_argmax_in_array(arr):
    """
    chooses a random index in a 1D-array from those where value is maximum
    """
    list_max_index = []
    max_v = -np.inf

    for i, v in enumerate(arr):
        if v > max_v:
            list_max_index = [i]
        elif v == max_v:
            list_max_index.append(i)

    random_index = np.random.choice(list_max_index)

    return random_index
