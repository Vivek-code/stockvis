import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_loader import download_data
from features import add_technical_indicators
from load_model_pkg import load_model_package
from model_utils import prepare_data_for_training

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def recalculate_metrics():
    base_dir = "models"
    if not os.path.exists(base_dir):
        print("No models directory found.")
        return

    tickers = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for ticker in tickers:
        ticker_path = os.path.join(base_dir, ticker)
        models = [m for m in os.listdir(ticker_path) if os.path.isdir(os.path.join(ticker_path, m))]
        
        if not models:
            continue
            
        print(f"\nProcessing {ticker}...")
        
        # We need data to evaluate. 
        # Strategy: Download same test range as training (or a fixed recent range).
        # Training used 2015-01-01 to 2024-01-01. Let's use that to be consistent with "Test Set"
        START_DATE = "2015-01-01"
        END_DATE = "2024-01-01"
        
        try:
            df = download_data(ticker, START_DATE, END_DATE)
            if df is None:
                print(f"  Skipping {ticker} (Data download failed)")
                continue

            df = add_technical_indicators(df)
            
            # For each model type
            for model_name in models:
                try:
                    pkg = load_model_package(model_name, ticker)
                    if pkg is None:
                        continue
                        
                    model = pkg['model']
                    scaler = pkg['scaler']
                    config = pkg['config']
                    
                    lookback = config['lookback']
                    feature_cols = config['features']
                    target_col = config.get('target', 'Close')
                    
                    # Prepare Data (Same logic as training)
                    # We need to recreate the exact test set used during training to match metrics,
                    # OR we just evaluate on the "test split" of this data.
                    # train_models.py used: test_size=0.2
                    
                    data_dict = prepare_data_for_training(
                        df, target_col=target_col, lookback=lookback, test_size=0.2
                    )
                    
                    X_test = data_dict['X_test']
                    y_test = data_dict['y_test']
                    
                    # Predict
                    y_pred_scaled = model.predict(X_test, verbose=0)
                    
                    # Inverse Transform
                    dummy_true = np.zeros((len(y_test), len(feature_cols)))
                    dummy_pred = np.zeros((len(y_pred_scaled), len(feature_cols)))
                    
                    target_idx = feature_cols.index(target_col)
                    
                    dummy_true[:, target_idx] = y_test.flatten()
                    dummy_pred[:, target_idx] = y_pred_scaled.flatten()
                    
                    y_true = scaler.inverse_transform(dummy_true)[:, target_idx]
                    y_pred = scaler.inverse_transform(dummy_pred)[:, target_idx]
                    
                    # Calculate Metrics
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    r2 = r2_score(y_true, y_pred)
                    mape = mean_absolute_percentage_error(y_true, y_pred)
                    
                    print(f"  {model_name.upper()} -> R2: {r2:.4f}, MAPE: {mape:.2f}%")
                    
                    # Update Config
                    config['mae'] = mae
                    config['rmse'] = rmse
                    config['r2'] = r2
                    config['mape'] = mape
                    
                    # Save Config
                    config_path = os.path.join(ticker_path, model_name, 'config.json')
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=4)
                        
                except Exception as e:
                    print(f"  Error updating {model_name} for {ticker}: {e}")
                    
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

if __name__ == "__main__":
    recalculate_metrics()
