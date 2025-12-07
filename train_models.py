import os
import json
import numpy as np
import pandas as pd
from data_loader import download_data
from features import add_technical_indicators
from model_utils import prepare_data_for_training, save_scaler
from model_definitions import create_lstm_model, create_gru_model, create_cnn_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration
TICKER = "RELIANCE.NS"
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"
LOOKBACK = 30
EPOCHS = 50 
BATCH_SIZE = 32

def train_and_save_models():
    # 1. Load and Preprocess Data
    print(f"Loading data for {TICKER}...")
    df = download_data(TICKER, START_DATE, END_DATE)
    if df is None:
        return
    
    print("Adding technical indicators...")
    df = add_technical_indicators(df)
    
    print("Preparing data for training...")
    data_dict = prepare_data_for_training(df, target_col='Close', lookback=LOOKBACK)
    
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']
    scaler = data_dict['scaler']
    feature_cols = data_dict['feature_cols']
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Define models to train
    models_to_train = {
        'lstm': create_lstm_model,
        'gru': create_gru_model,
        'cnn': create_cnn_model
    }
    
    base_dir = "models"
    os.makedirs(base_dir, exist_ok=True)
    
    results = {}

    for name, model_fn in models_to_train.items():
        print(f"\nTraining {name.upper()} model...")
        model = model_fn(input_shape)
        
        model_dir = os.path.join(base_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            # ModelCheckpoint(filepath=os.path.join(model_dir, f'{name}_best.keras'), save_best_only=True)
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save Model
        model_path = os.path.join(model_dir, f'{name}_model.keras')
        model.save(model_path)
        print(f"Saved {name} model to {model_path}")
        
        # Save Scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        save_scaler(scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
        
        # Evaluate
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        # Manually compute RMSE
        preds = model.predict(X_test)
        rmse = np.sqrt(np.mean((preds.flatten() - y_test) ** 2))
        
        print(f"{name.upper()} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        # Save Config
        config = {
            "name": f"{name.upper()}_v1",
            "model_type": name.upper(),
            "lookback": LOOKBACK,
            "features": feature_cols,
            "target": "Close",
            "train_start": START_DATE,
            "train_end": END_DATE,
            "metrics": {
                "mae": float(mae),
                "rmse": float(rmse),
                "val_loss": float(loss)
            }
        }
        
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
            
        results[name] = {"mae": mae, "rmse": rmse}

    print("\nTraining Complete!")
    print("Results:", results)

if __name__ == "__main__":
    train_and_save_models()
