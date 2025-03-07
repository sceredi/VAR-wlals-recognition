"""Helper functions used to save and load pickle files."""

import pickle as pkl
from typing import List

import numpy as np


def save_array(arr: List | np.ndarray, path: str) -> None:
    """Save a numpy array to a pickle file.

    Parameters
    ----------
    arr : List | np.ndarray
        The array to save.
    path : str
        The path to save the array to.
    """
    file = open(path, "wb")
    pkl.dump(arr, file)
    file.close()


def load_array(path: str) -> np.ndarray:
    """Load a numpy array from a pickle file.

    Parameters
    ----------
    path : str
        The path to load the array from.

    Returns
    -------
    np.ndarray
        The loaded numpy array.
    """
    file = open(path, "rb")
    arr = pkl.load(file)
    file.close()
    return np.array(arr)
