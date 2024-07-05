import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import numpy as np


def compute_user_similarity(df, method='cosine'):
    """Computes user similarity based on ratings using the specified method."""
    # Aggregate duplicate (user_id, item_id) pairs by taking the mean rating
    df = df.groupby(['user_id', 'item_id']).rating.mean().reset_index()

    user_ratings = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

    if method == 'cosine':
        similarity_matrix = cosine_similarity(user_ratings)
    elif method == 'euclidean':
        similarity_matrix = 1 / (1 + pairwise_distances(user_ratings, metric='euclidean'))
    elif method == 'pearson':
        similarity_matrix = np.corrcoef(user_ratings)
    else:
        raise ValueError("Invalid similarity computation method. Choose 'cosine', 'euclidean', or 'pearson'.")

    return pd.DataFrame(similarity_matrix, index=user_ratings.index, columns=user_ratings.index)


def compute_item_similarity(df, method='cosine'):
    """Computes item similarity based on ratings using the specified method."""
    # Aggregate duplicate (user_id, item_id) pairs by taking the mean rating
    df = df.groupby(['user_id', 'item_id']).rating.mean().reset_index()

    item_ratings = df.pivot(index='item_id', columns='user_id', values='rating').fillna(0)

    if method == 'cosine':
        similarity_matrix = cosine_similarity(item_ratings)
    elif method == 'euclidean':
        similarity_matrix = 1 / (1 + pairwise_distances(item_ratings, metric='euclidean'))
    elif method == 'pearson':
        similarity_matrix = np.corrcoef(item_ratings)
    else:
        raise ValueError("Invalid similarity computation method. Choose 'cosine', 'euclidean', or 'pearson'.")

    return pd.DataFrame(similarity_matrix, index=item_ratings.index, columns=item_ratings.index)

