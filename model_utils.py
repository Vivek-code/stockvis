import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

def create_sliding_window(data, lookback, target_col_idx):
    """
    Creates sliding window sequences for time series forecasting.
    
    Args:
        data (np.array): Scaled data of shape (num_samples, num_features).
        lookback (int): Number of past days to include in input.
        target_col_idx (int): Index of the target column (e.g., adjusted close) to predict.
    
    Returns:
        X (np.array): Input sequences of shape (num_samples - lookback, lookback, num_features)
        y (np.array): Target values of shape (num_samples - lookback,)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, target_col_idx])
    
    return np.array(X), np.array(y)

def save_scaler(scaler, path):
    """Saves the scaler object using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path):
    """Loads a scaler object."""
    return joblib.load(path)

def prepare_data_for_training(df, target_col='Close', lookback=30, test_size=0.2):
    """
    Full pipeline: Split -> Scale -> Sliding Window.
    
    Args:
        df (pd.DataFrame): Dataframe with features.
        target_col (str): Name of the target column.
        lookback (int): Window size.
        test_size (float): Fraction of data to use for testing (latest data).
        
    Returns:
        dict: Contains X_train, y_train, X_test, y_test, scaler, feature_cols
    """
    # 1. Split Data (Time-based split, no shuffle)
    train_size = int(len(df) * (1 - test_size))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    feature_cols = df.columns.tolist()
    target_idx = feature_cols.index(target_col)
    
    # 2. Scale Data
    # Fit scaler ONLY on training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df)
    
    train_scaled = scaler.transform(train_df)
    test_scaled = scaler.transform(test_df)
    
    # 3. Create Sliding Windows
    X_train, y_train = create_sliding_window(train_scaled, lookback, target_idx)
    X_test, y_test = create_sliding_window(test_scaled, lookback, target_idx)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'lookback': lookback
    }
