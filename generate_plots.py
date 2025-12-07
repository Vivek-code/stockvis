import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from data_loader import download_data
from features import add_technical_indicators
from load_model_pkg import load_model_package
from model_utils import prepare_data_for_training

def generate_comparison_plots(ticker="RELIANCE.NS"):
    print(f"Generating comparison plots for {ticker}...")
    
    # 1. Load Data
    df = download_data(ticker, "2023-01-01", "2024-01-01") # Test period
    if df is None:
        return
    
    df = add_technical_indicators(df)
    
    # 2. Load Models
    models = {}
    try:
        models['LSTM'] = load_model_package("lstm", ticker)
        models['GRU'] = load_model_package("gru", ticker)
        models['CNN'] = load_model_package("cnn", ticker)
    except Exception as e:
        print(f"Error loading models for {ticker}: {e}")
        return

    # Use parameters from one config (assuming consistent)
    config = models['LSTM']['config']
    lookback = config['lookback']
    feature_cols = config['features']
    scaler = models['LSTM']['scaler'] # Reuse scaler (Approximation if models differ) but strictly should use own.
    
    # Note: Using the scaler from training (Reliance) for other stocks is NOT mathematically correct 
    # without re-fitting, but for a "Show me something" demo, we might just re-fit the scaler on the new stock data
    # or accept the discrepancy. 
    # BETTER: Fit a new scaler on the new stock's data for visualization purposes so the range matches.
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(df[feature_cols])

    # Prepare Data
    data = scaler.transform(df[feature_cols])
    X, y = [], []
    target_idx = feature_cols.index("Close")
    
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, target_idx])
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("Not enough data.")
        return

    # 3. Predict & Plot
    plt.figure(figsize=(12, 6))
    
    # Inverse transform y_true
    dummy = np.zeros((len(y), len(feature_cols)))
    dummy[:, target_idx] = y
    y_true_prices = scaler.inverse_transform(dummy)[:, target_idx]
    
    plt.plot(df.index[lookback:], y_true_prices, label="Actual", color='black', linewidth=2)
    
    metrics = {}
    
    colors = {'LSTM': '#0d6efd', 'GRU': '#198754', 'CNN': '#ffc107'}
    
    for name, pkg in models.items():
        model = pkg['model']
        # We should strictly use the scaler belonging to the model if we wanted "Transfer Learning" 
        # but since we re-scaled the input X to (0,1) based on current data, the model might behave okay-ish 
        # or wildly wrong. 
        # For this demo, let's use the local X (0-1) and see what the model outputs.
        
        preds = model.predict(X, verbose=0)
        
        # Inverse transform preds
        dummy_pred = np.zeros((len(preds), len(feature_cols)))
        dummy_pred[:, target_idx] = preds.flatten()
        pred_prices = scaler.inverse_transform(dummy_pred)[:, target_idx]
        
        plt.plot(df.index[lookback:], pred_prices, label=name, color=colors.get(name, 'blue'), alpha=0.7)
        
        # Calc simplistic RMSE
        rmse = np.sqrt(np.mean((y_true_prices - pred_prices)**2))
        metrics[name] = rmse
        
    plt.title(f"Model Comparison: Actual vs Predicted ({ticker})")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save Plot
    save_path = f"static/images/comparison_{ticker}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved plot to {save_path}")
    return metrics

if __name__ == "__main__":
    tickers = [
        "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "TATAMOTORS.NS",
        "INFY.NS", "LT.NS", "NTPC.NS", "ADANIENT.NS", "TATASTEEL.NS"
    ]
    for t in tickers:
        try:
            generate_comparison_plots(t)
        except Exception as e:
            print(f"Failed to generate for {t}: {e}")
