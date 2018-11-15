import numpy as np

import enlighten
import json
import os

progress_manager = enlighten.get_manager()


def makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


def format_json(dict_):
    return json.dumps(dict_, indent=4, sort_keys=True)


def load_json(path):
    with open(path, 'rb') as f:
        return json.load(f)


def save_json(dict_, path):
    with open(path, 'w+') as f:
        json.dump(dict_, f, indent=4, sort_keys=True)
