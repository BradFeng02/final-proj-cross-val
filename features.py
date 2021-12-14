"""
defines constants, and function for extracting features
"""

from typing import Tuple
import numpy as np

N_FEAT = 7

def find_feats(s: str, arr: np.ndarray, i: int):
    """finds features of s, puts them in arr[i][]."""

    # TODO put features here
    # example: arr[i][0] = feature 1, arr[i][1] = feature 2, etc.
    # s is the whole sample

#enddef

def standardize_feats(samples: np.ndarray, n_train: int):
    feat_range = np.zeros((N_FEAT, 2))  # (min, max) for standardizing
    for i in range(2 * n_train):
        for j in range(N_FEAT):
            if samples[i, j] < feat_range[j, 0]:
                feat_range[j, 0] = samples[i, j]
            elif samples[i, j] > feat_range[j, 1]:
                feat_range[j, 1] = samples[i, j]

    for i in range(2 * n_train):  # standardize to [-1 , 1]
        for j in range(N_FEAT):
            if feat_range[j, 1] == feat_range[j, 0]:
                samples[i, j] = 0
            else:
                samples[i, j] = (samples[i, j] - feat_range[j, 0]) / \
                    (feat_range[j, 1] - feat_range[j, 0]) * 2 - 1

def make_input_arrs(n_train: int) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.zeros((2 * n_train, N_FEAT), dtype=np.float64)
    labels = np.zeros(2 * n_train, dtype=np.float64)
    return (samples, labels)