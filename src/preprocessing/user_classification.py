import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import numpy as np

def classify_users(df, method='kmeans', n_clusters=3):
    """Classifies users into clusters based on demographic and behavioral data."""
    # Feature engineering
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 50, 65, np.inf], labels=[0, 1, 2, 3, 4])
    df['is_high_spender'] = df['price'] > df['price'].median()

    # Encode categorical variables
    le_location = LabelEncoder()
    df['location'] = le_location.fit_transform(df['location'])

    # Selecting features for clustering
    features = df[['age_group', 'gender', 'location', 'is_high_spender']]

    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Check for NaN values and handle them
    if np.any(np.isnan(features)):
        print("NaN values found in features. Handling NaN values...")
        features = np.nan_to_num(features)

    # Dimensionality reduction
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    # Clustering
    if method == 'kmeans':
        clustering_model = KMeans(n_clusters=n_clusters, random_state=0)
    elif method == 'agglomerative':
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'dbscan':
        clustering_model = DBSCAN(eps=0.5, min_samples=5)
    else:
        raise ValueError("Invalid clustering method. Choose 'kmeans', 'agglomerative', or 'dbscan'.")

    df['user_class'] = clustering_model.fit_predict(features_pca)
    return df


