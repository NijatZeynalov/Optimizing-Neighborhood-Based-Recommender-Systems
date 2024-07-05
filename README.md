# Optimizing Neighborhood-Based Recommender Systems with User Classification and Advanced Rating Normalization Techniques

The objective of this project is to develop and evaluate a neighborhood-based recommender system that utilizes user classification, advanced rating normalization techniques, correlation-based similarity computation methods, and optimized neighborhood selection methods to improve recommendation accuracy and user satisfaction.


## Usage

#### Running the Pipeline

You can run the entire pipeline using the main.py script. Make sure to provide the configuration file path.

```bash
python main.py --config config.json
```
It should return result something like that:

```python
Evaluation Results: { 'mae': 0.8321, 'rmse': 0.9563, 'precision': 0.745, 'recall': 0.812, 'f1_score': 0.777 }

Saving evaluation results to data/processed/evaluation_results.csv...
```

#### Tuning Hyperparameters

To tune the hyperparameters, use the --tune flag.

```bash
python main.py --config config.json --tune
```


## Key Techniques and Methods
* User Classification -> helps to segment users into different clusters based on their behavior and preferences. This project uses clustering algorithms like KMeans to classify users into different groups. These groups can then be used to improve the accuracy of recommendations by considering the preferences of similar users.

* Rating Normalization -> is crucial to ensure that the rating scales of different users are comparable. This project explores multiple normalization techniques:

Mean-Centering, Z-Score Normalization, Robust Scaling, Hybrid Normalization.


* Similarity Weight Computation -> computing the similarity between users or items is key to neighborhood-based recommender systems. This project focuses on correlation-based methods:

Pearson Correlation, Cosine Similarity, Jaccard Similarity.

* Neighborhood Selection -> selecting the right neighbors is crucial for generating accurate recommendations. This project explores different neighborhood selection methods:

Top-N Nearest Neighbors, Threshold-Based Selection, Hybrid Selection: Combines multiple selection criteria for improved performance.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
