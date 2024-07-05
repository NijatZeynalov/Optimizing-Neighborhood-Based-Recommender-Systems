import pandas as pd
import numpy as np


def select_top_n_neighbors(similarity_matrix, n=5):
    """Selects the top N neighbors for each user/item."""
    top_n_neighbors = {}
    for index in similarity_matrix.index:
        top_n_neighbors[index] = similarity_matrix.loc[index].nlargest(n + 1).iloc[1:].index.tolist()
    return top_n_neighbors


def select_threshold_neighbors(similarity_matrix, threshold=0.5):
    """Selects neighbors with similarity weights above a certain threshold."""
    threshold_neighbors = {}
    for index in similarity_matrix.index:
        threshold_neighbors[index] = similarity_matrix.loc[index][
            similarity_matrix.loc[index] > threshold].index.tolist()
    return threshold_neighbors


def select_negative_neighbors(similarity_matrix, threshold=0):
    """Excludes negatively correlated neighbors."""
    negative_neighbors = {}
    for index in similarity_matrix.index:
        negative_neighbors[index] = similarity_matrix.loc[index][
            similarity_matrix.loc[index] > threshold].index.tolist()
    return negative_neighbors


def select_neighbors(similarity_matrix, method='top_n', n=5, threshold=0.5):
    """Selects neighbors based on the specified method."""
    if method == 'top_n':
        return select_top_n_neighbors(similarity_matrix, n=n)
    elif method == 'threshold':
        return select_threshold_neighbors(similarity_matrix, threshold=threshold)
    elif method == 'negative':
        return select_negative_neighbors(similarity_matrix, threshold=threshold)
    else:
        raise ValueError("Invalid neighborhood selection method. Choose 'top_n', 'threshold', or 'negative'.")

