import pandas as pd
import json
import logging

def save_results(results, file_path):
    """Saves the results to a CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)
    logging.info(f"Results saved to {file_path}")

def load_config(file_path):
    """Loads configuration settings from a JSON file."""
    with open(file_path, 'r') as file:
        config = json.load(file)
    logging.info(f"Configuration loaded from {file_path}")
    return config

def setup_logging(log_file):
    """Sets up logging to the specified file."""
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Logging setup complete")

def log_evaluation_results(results):
    """Logs the evaluation results."""
    logging.info("Evaluation Results:")
    for key, value in results.items():
        logging.info(f"{key}: {value}")

def load_data(file_path):
    """Loads data from a CSV file."""
    data = pd.read_csv(file_path)
    logging.info(f"Data loaded from {file_path}")
    return data

def save_data(data, file_path):
    """Saves data to a CSV file."""
    data.to_csv(file_path, index=False)
    logging.info(f"Data saved to {file_path}")


