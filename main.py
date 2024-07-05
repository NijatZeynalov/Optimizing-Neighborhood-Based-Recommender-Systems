import os
import argparse
import pandas as pd
import numpy as np
from src.preprocessing.data_preprocessing import load_data, preprocess_data
from src.preprocessing.user_classification import classify_users
from src.preprocessing.rating_normalization import mean_centering, z_score_normalization, robust_scaling, \
    hybrid_normalization
from src.recommendation.similarity_computation import compute_user_similarity, compute_item_similarity
from src.recommendation.neighborhood_selection import select_neighbors
from src.recommendation.recommendation_algorithm import predict_ratings_user_based, predict_ratings_item_based, \
    hybrid_prediction
from src.evaluation.evaluation import evaluate_predictions, save_evaluation_results
from utils.helper_functions import setup_logging, save_results, load_config, log_evaluation_results


def run_pipeline(config):
    data_file = config['data_file']
    processed_dir = config['processed_dir']
    os.makedirs(processed_dir, exist_ok=True)

    # Load and preprocess data
    data = load_data(data_file)
    data = preprocess_data(data)

    # Classify users
    data = classify_users(data, method=config['classification_method'], n_clusters=config['n_clusters'])

    # Apply rating normalization
    if config['normalization_method'] == 'mean_centering':
        data = mean_centering(data)
    elif config['normalization_method'] == 'z_score':
        data = z_score_normalization(data)
    elif config['normalization_method'] == 'robust_scaling':
        data = robust_scaling(data)
    elif config['normalization_method'] == 'hybrid':
        data = hybrid_normalization(data)

    # Compute similarity
    if config['similarity_method'] == 'user':
        similarity_matrix = compute_user_similarity(data, method=config['similarity_metric'])
    elif config['similarity_method'] == 'item':
        similarity_matrix = compute_item_similarity(data, method=config['similarity_metric'])

    # Select neighbors
    neighbors = select_neighbors(similarity_matrix, method=config['neighborhood_method'], n=config['n_neighbors'],
                                 threshold=config['threshold'])

    # Predict ratings
    if config['recommendation_method'] == 'user_based':
        predicted_ratings = predict_ratings_user_based(data, neighbors)
    elif config['recommendation_method'] == 'item_based':
        predicted_ratings = predict_ratings_item_based(data, neighbors)
    elif config['recommendation_method'] == 'hybrid':
        user_based_predictions = predict_ratings_user_based(data, neighbors)
        item_based_predictions = predict_ratings_item_based(data, neighbors)
        predicted_ratings = hybrid_prediction(user_based_predictions, item_based_predictions, alpha=config['alpha'])

    predicted_ratings_df = pd.DataFrame.from_dict(predicted_ratings, orient='index', columns=['predicted_rating'])
    predicted_ratings_df.index = pd.MultiIndex.from_tuples(predicted_ratings_df.index, names=['user_id', 'item_id'])

    # Ensure no duplicate indices
    predicted_ratings_df = predicted_ratings_df[~predicted_ratings_df.index.duplicated(keep='first')]

    # Convert multi-index to columns
    predicted_ratings_df.reset_index(inplace=True)
    true_ratings = data[['user_id', 'item_id', 'rating']].reset_index(drop=True)

    # Align the indices of true and predicted ratings
    true_ratings.set_index(['user_id', 'item_id'], inplace=True)
    predicted_ratings_df.set_index(['user_id', 'item_id'], inplace=True)

    common_index = true_ratings.index.intersection(predicted_ratings_df.index)
    true_ratings = true_ratings.loc[common_index]
    predicted_ratings_df = predicted_ratings_df.loc[common_index]

    true_ratings.set_index(['user_id', 'item_id'], inplace=True)
    predicted_ratings_df.set_index(['user_id', 'item_id'], inplace=True)

    evaluation_results = evaluate_predictions(true_ratings, predicted_ratings_df)
    save_evaluation_results(evaluation_results, os.path.join(processed_dir, 'evaluation_results.csv'))

    log_evaluation_results(evaluation_results)

    return evaluation_results


def tune_parameters(config):
    normalization_methods = config['normalization_methods']
    similarity_methods = config['similarity_methods']
    neighborhood_methods = config['neighborhood_methods']
    n_neighbors_options = config['n_neighbors_options']
    alphas = config['alphas']

    best_mae = float('inf')
    best_rmse = float('inf')
    best_params = {}
    results = []

    for norm_method in normalization_methods:
        for sim_method in similarity_methods:
            for neigh_method in neighborhood_methods:
                for n_neighbors in n_neighbors_options:
                    for alpha in alphas:
                        config.update({
                            'normalization_method': norm_method,
                            'similarity_method': sim_method,
                            'neighborhood_method': neigh_method,
                            'n_neighbors': n_neighbors,
                            'alpha': alpha
                        })
                        evaluation_results = run_pipeline(config)
                        mae, rmse = evaluation_results['mae'], evaluation_results['rmse']
                        results.append({
                            'normalization_method': norm_method,
                            'similarity_method': sim_method,
                            'neighborhood_method': neigh_method,
                            'n_neighbors': n_neighbors,
                            'alpha': alpha,
                            'mae': mae,
                            'rmse': rmse
                        })
                        if mae < best_mae:
                            best_mae = mae
                            best_rmse = rmse
                            best_params = {
                                'normalization_method': norm_method,
                                'similarity_method': sim_method,
                                'neighborhood_method': neigh_method,
                                'n_neighbors': n_neighbors,
                                'alpha': alpha
                            }

    save_results(results, os.path.join(config['processed_dir'], 'tuning_results.csv'))
    save_results([best_params], os.path.join(config['processed_dir'], 'best_parameters.csv'))

    print("Best Parameters:")
    print(best_params)
    print(f"Best MAE: {best_mae}")
    print(f"Best RMSE: {best_rmse}")


def main():
    parser = argparse.ArgumentParser(description='Run Recommender System Pipeline')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters')

    args = parser.parse_args()

    config = load_config(args.config)

    setup_logging(config['log_file'])

    if args.tune:
        tune_parameters(config)
    else:
        evaluation_results = run_pipeline(config)
        print("Evaluation Results:")
        print(evaluation_results)


if __name__ == "__main__":
    main()
