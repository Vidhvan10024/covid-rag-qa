"""Crash-safe checkpoint helpers.

Colab can kill the runtime at any moment, including in the middle of writing a file.
Every checkpoint is written to a temporary file first and then renamed into place, so a
checkpoint on disk is always either the old complete version or the new complete version.
"""
import json
import os

import numpy as np


def save_json(obj, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_json(path: str):
    """Returns the object, or None if the file is missing or unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_npy(arr, path: str):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_npy(path: str):
    try:
        return np.load(path)
    except (OSError, ValueError):
        return None
