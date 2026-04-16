"""
Standalone script to retrain transformer models for specific tickers.
Deletes existing transformer folders and retrains with improved hyperparameters.
"""
import os
import json
import shutil
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_loader import download_data
from features import add_technical_indicators
from model_utils import prepare_data_for_training, save_scaler
from model_definitions import create_transformer_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configuration
START_DATE = "2020-01-01"
END_DATE = "2026-04-16"
LOOKBACK = 30
EPOCHS = 150
BATCH_SIZE = 16
TARGET_COL = 'Close'
TEST_SIZE = 0.2

# Tickers to retrain (negative R² or underperforming)
RETRAIN_TICKERS = [
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "NTPC.NS",
    "ADANIENT.NS",
    "RELIANCE.NS",
    "INFY.NS",
    "LT.NS",
]

BASE_MODEL_DIR = "models"

def save_model_package(model, scaler, config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, 'transformer_model.keras'))
    from model_utils import save_scaler as _save
    _save(scaler, os.path.join(save_dir, 'scaler.pkl'))
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def retrain():
    for ticker in RETRAIN_TICKERS:
        print(f"\n{'='*60}")
        print(f"  RETRAINING TRANSFORMER for {ticker}")
        print(f"{'='*60}\n")

        save_dir = os.path.join(BASE_MODEL_DIR, ticker, "transformer")

        # Delete existing transformer model for this ticker
        if os.path.exists(save_dir):
            print(f"  Deleting old model at {save_dir}...")
            shutil.rmtree(save_dir)

        try:
            # 1. Download data
            df = download_data(ticker, START_DATE, END_DATE)
            if df is None:
                print(f"  Skipping {ticker} — download failed.")
                continue

            # 2. Add features
            df = add_technical_indicators(df)

            # 3. Prepare data
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
            print(f"  Input shape: {input_shape}, Train samples: {len(X_train)}, Test samples: {len(X_test)}")

            # 4. Create and train model
            model = create_transformer_model(input_shape)

            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )

            # 5. Evaluate
            y_pred = model.predict(X_test)

            dummy_true = np.zeros((len(y_test), len(feature_cols)))
            dummy_pred = np.zeros((len(y_pred), len(feature_cols)))
            target_idx = feature_cols.index(TARGET_COL)

            dummy_true[:, target_idx] = y_test.flatten()
            dummy_pred[:, target_idx] = y_pred.flatten()

            y_true_rescaled = scaler.inverse_transform(dummy_true)[:, target_idx]
            y_pred_rescaled = scaler.inverse_transform(dummy_pred)[:, target_idx]

            mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
            rmse = np.sqrt(mean_squared_error(y_true_rescaled, y_pred_rescaled))
            r2 = r2_score(y_true_rescaled, y_pred_rescaled)

            mask = y_true_rescaled != 0
            mape = np.mean(np.abs((y_true_rescaled[mask] - y_pred_rescaled[mask]) / y_true_rescaled[mask])) * 100

            print(f"\n  RESULTS — MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")

            # 6. Save
            config = {
                'ticker': ticker,
                'model_type': 'transformer',
                'features': feature_cols,
                'target': TARGET_COL,
                'lookback': LOOKBACK,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            save_model_package(model, scaler, config, save_dir)
            print(f"  Saved to {save_dir}\n")

        except Exception as e:
            print(f"  ERROR training {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*60)
    print("  ALL RETRAINING COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    retrain()
