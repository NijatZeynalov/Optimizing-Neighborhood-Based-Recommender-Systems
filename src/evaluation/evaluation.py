import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score


def evaluate_predictions(true_ratings, predicted_ratings, threshold=3.5):
    """Evaluates the predicted ratings using MAE, RMSE, Precision, Recall, and F1-Score."""
    true_ratings = true_ratings.copy()
    predicted_ratings = predicted_ratings.copy()

    true_ratings.set_index(['user_id', 'item_id'], inplace=True)
    predicted_ratings.set_index(['user_id', 'item_id'], inplace=True)

    common_index = true_ratings.index.intersection(predicted_ratings.index)
    true_ratings = true_ratings.loc[common_index]
    predicted_ratings = predicted_ratings.loc[common_index]

    true_values = true_ratings['rating']
    predicted_values = predicted_ratings['predicted_rating']

    mae = mean_absolute_error(true_values, predicted_values)
    rmse = mean_squared_error(true_values, predicted_values, squared=False)

    # Binarize ratings for classification metrics
    true_binary = (true_values >= threshold).astype(int)
    predicted_binary = (predicted_values >= threshold).astype(int)

    precision = precision_score(true_binary, predicted_binary, zero_division=0)
    recall = recall_score(true_binary, predicted_binary, zero_division=0)
    f1 = f1_score(true_binary, predicted_binary, zero_division=0)

    return {
        'mae': mae,
        'rmse': rmse,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def save_evaluation_results(results, file_path):
    """Saves the evaluation results to a CSV file."""
    df = pd.DataFrame([results])
    df.to_csv(file_path, index=False)


