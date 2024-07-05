import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def mean_centering(df):
    """Applies mean-centering normalization to the ratings."""
    df['rating'] = df.groupby('user_id')['rating'].transform(lambda x: x - x.mean())
    return df


def z_score_normalization(df):
    """Applies Z-score normalization to the ratings."""
    df['rating'] = df.groupby('user_id')['rating'].transform(lambda x: (x - x.mean()) / x.std())
    return df


def robust_scaling(df):
    """Applies robust scaling to the ratings to handle outliers."""
    df['rating'] = df.groupby('user_id')['rating'].transform(
        lambda x: (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25)))
    return df


def hybrid_normalization(df):
    """Combines multiple normalization techniques for better handling of diverse data distributions."""
    df_mean_centered = mean_centering(df.copy())
    df_z_score = z_score_normalization(df.copy())
    df_robust = robust_scaling(df.copy())

    # Combining the results by averaging the normalized ratings
    df['rating'] = (df_mean_centered['rating'] + df_z_score['rating'] + df_robust['rating']) / 3
    return df



