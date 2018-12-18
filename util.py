import numpy as np

import enlighten
import json
import os

progress_manager = enlighten.get_manager()


def makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


def numpy_to_builtin(x):
    if type(x) == dict:
        return { k: numpy_to_builtin(v) for k, v in x.items() }
    elif type(x) in [list, tuple]:
        return [numpy_to_builtin(v) for v in x]
    if type(x).__module__ == np.__name__:
        return np.asscalar(x)
    return x


def format_json(dict_):
    return json.dumps(numpy_to_builtin(dict_), indent=4, sort_keys=True)


def load_json(path):
    with open(path, 'r+') as f:
        return json.load(f)


def save_json(dict_, path):
    with open(path, 'w+') as f:
        json.dump(numpy_to_builtin(dict_), f, indent=4, sort_keys=True)

