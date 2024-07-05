import pandas as pd
import numpy as np


def predict_ratings_user_based(df, user_top_n_neighbors):
    """Predicts ratings using a user-based collaborative filtering approach."""
    predicted_ratings = {}
    for user in df['user_id'].unique():
        neighbors = user_top_n_neighbors.get(user, [])
        neighbor_ratings = df[df['user_id'].isin(neighbors)]
        user_ratings = df[df['user_id'] == user]

        for item in df['item_id'].unique():
            ratings = neighbor_ratings[neighbor_ratings['item_id'] == item]['rating']
            if not ratings.empty:
                predicted_ratings[(user, item)] = ratings.mean()
            else:
                if not user_ratings[user_ratings['item_id'] == item].empty:
                    predicted_ratings[(user, item)] = user_ratings[user_ratings['item_id'] == item]['rating'].mean()
                else:
                    predicted_ratings[(user, item)] = df[df['item_id'] == item]['rating'].mean()
    return predicted_ratings


def predict_ratings_item_based(df, item_top_n_neighbors):
    """Predicts ratings using an item-based collaborative filtering approach."""
    predicted_ratings = {}
    for user in df['user_id'].unique():
        user_ratings = df[df['user_id'] == user]

        for item in df['item_id'].unique():
            neighbors = item_top_n_neighbors.get(item, [])
            neighbor_ratings = df[(df['item_id'].isin(neighbors)) & (df['user_id'] == user)]
            if not neighbor_ratings.empty:
                predicted_ratings[(user, item)] = neighbor_ratings['rating'].mean()
            else:
                if not user_ratings[user_ratings['item_id'] == item].empty:
                    predicted_ratings[(user, item)] = user_ratings[user_ratings['item_id'] == item]['rating'].mean()
                else:
                    predicted_ratings[(user, item)] = df[df['item_id'] == item]['rating'].mean()
    return predicted_ratings


def hybrid_prediction(user_based_predictions, item_based_predictions, alpha=0.5):
    """Combines user-based and item-based predictions using a weighted average."""
    hybrid_predictions = {}
    for key in user_based_predictions:
        user_based = user_based_predictions[key]
        item_based = item_based_predictions.get(key, user_based)  # Default to user-based if item-based is not available

        # Ensure both predictions are numeric values
        try:
            user_based = float(user_based)
        except ValueError:
            print(f"Skipping invalid user_based value for key {key}: {user_based}")
            continue

        try:
            item_based = float(item_based)
        except ValueError:
            print(f"Skipping invalid item_based value for key {key}: {item_based}")
            continue

        hybrid_predictions[key] = alpha * user_based + (1 - alpha) * item_based

    return hybrid_predictions


