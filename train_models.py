import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_loader import download_data
from features import add_technical_indicators
from model_utils import prepare_data_for_training, save_scaler
from model_definitions import create_lstm_model, create_gru_model, create_cnn_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"
LOOKBACK = 30
EPOCHS = 20 # Efficient training for 10 tickers
BATCH_SIZE = 32
TARGET_COL = 'Close'
TEST_SIZE = 0.2

def save_model_package(model, scaler, config, save_dir, model_name):
    os.makedirs(save_dir, exist_ok=True)
    
    # Save Model
    model_path = os.path.join(save_dir, f'{model_name}_model.keras')
    model.save(model_path)
    
    # Save Scaler
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    save_scaler(scaler, scaler_path)
    
    # Save Config
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def train_and_save_models():
    # Define models to train
    models_to_train = {
        'lstm': create_lstm_model,
        'gru': create_gru_model,
        'cnn': create_cnn_model
    }
    
    base_model_dir = "models"
    
    tickers = [
        "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "TATAMOTORS.NS",
        "INFY.NS", "LT.NS", "NTPC.NS", "ADANIENT.NS", "TATASTEEL.NS"
    ]

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Starting training pipeline for {ticker}...")
        print(f"{'='*50}\n")
        
        try:
            # 1. Load Data
            df = download_data(ticker, START_DATE, END_DATE)
            if df is None:
                print(f"Skipping {ticker} due to download failure.")
                continue

            # 2. Add Features
            df = add_technical_indicators(df)
            
            # 3. Prepare Data
            data_dict = prepare_data_for_training(
                df, target_col=TARGET_COL, lookback=LOOKBACK, test_size=TEST_SIZE
            )
            
            X_train = data_dict['X_train']
            y_train = data_dict['y_train']
            X_test = data_dict['X_test']
            y_test = data_dict['y_test']
            scaler = data_dict['scaler']
            feature_cols = data_dict['feature_cols']
            
            input_shape = (LOOKBACK, len(feature_cols))
            
            # 4. Train Models
            for name, model_fn in models_to_train.items():
                save_dir = os.path.join(base_model_dir, ticker, name)
                if os.path.exists(os.path.join(save_dir, 'config.json')):
                    print(f"Skipping {name} for {ticker} (Already trained)")
                    continue

                print(f"\nTraining {name.upper()} model for {ticker}...")
                model = model_fn(input_shape)
                
                # Callbacks
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[early_stopping],
                    verbose=1
                )
                
                # 5. Evaluation
                loss = model.evaluate(X_test, y_test, verbose=0)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Inverse transform
                dummy_true = np.zeros((len(y_test), len(feature_cols)))
                dummy_pred = np.zeros((len(y_pred), len(feature_cols)))
                
                target_idx = feature_cols.index(TARGET_COL)
                
                dummy_true[:, target_idx] = y_test.flatten()
                dummy_pred[:, target_idx] = y_pred.flatten()
                
                y_true_rescaled = scaler.inverse_transform(dummy_true)[:, target_idx]
                y_pred_rescaled = scaler.inverse_transform(dummy_pred)[:, target_idx]
                
                mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
                rmse = np.sqrt(mean_squared_error(y_true_rescaled, y_pred_rescaled))
                
                print(f"{name.upper()} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
                
                # 6. Save Model Package
                # Structure: models/TICKER/MODEL_NAME
                config = {
                    'ticker': ticker,
                    'model_type': name,
                    'features': feature_cols,
                    'target': TARGET_COL,
                    'lookback': LOOKBACK,
                    'mae': mae,
                    'rmse': rmse,
                    'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                save_dir = os.path.join(base_model_dir, ticker, name)
                save_model_package(model, scaler, config, save_dir, name)
                print(f"Saved {name} package for {ticker} to {save_dir}")

        except Exception as e:
            print(f"Error training for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\nAll training completed!")

if __name__ == "__main__":
    train_and_save_models()
