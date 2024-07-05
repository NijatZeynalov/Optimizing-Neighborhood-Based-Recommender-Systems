import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocesses the data for further analysis."""
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Fill missing values
    df.fillna({
        'rating': df['rating'].median(),
        'price': df['price'].median(),
        'brand': 'Unknown',
        'age': df['age'].median()
    }, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    df['category'] = le.fit_transform(df['category'])
    df['brand'] = le.fit_transform(df['brand'])
    df['location'] = le.fit_transform(df['location'])
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # Scale numerical features
    scaler = StandardScaler()
    df[['price', 'age']] = scaler.fit_transform(df[['price', 'age']])
    return df

