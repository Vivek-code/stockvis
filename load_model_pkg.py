import os
import json
import joblib
import numpy as np
from tensorflow.keras.models import load_model

def load_model_package(model_name, ticker="RELIANCE.NS"):
    """
    Loads a model package (model, scaler, config) by name and ticker.
    Structure: models/<TICKER>/<MODEL_NAME>/
    """
    base_dir = "models"
    model_dir = os.path.join(base_dir, ticker, model_name)
    
    # Absolute path check
    if not os.path.isabs(model_dir):
        # Assuming script is run from project root
        model_dir = os.path.abspath(model_dir)
    
    # 1. Load Config
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found for {model_name} at {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # 2. Load Scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found for {model_name}")
        
    scaler = joblib.load(scaler_path)
    
    # 3. Load Model
    model_path = os.path.join(model_dir, f"{model_name}_model.keras")
    if not os.path.exists(model_path):
        # Fallback to .h5 if .keras doesn't exist
        model_path = os.path.join(model_dir, f"{model_name}_model.h5")
        
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"Model file not found for {model_name}")

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    return {
        'model': model,
        'scaler': scaler,
        'config': config
    }

# Cache for ensemble so we don't load repeatedly from disk
_ensemble_cache = {}

def ensemble_predict(ticker, X_input, scaler):
    """
    Computes a weighted ensemble prediction across available models.
    Returns the final predicted price as a float.
    """
    weights = {'transformer': 0.35, 'gru': 0.25, 'lstm': 0.25, 'cnn': 0.15}
    loaded_models = {}
    feature_cols = None
    target_name = None
    
    for m in weights.keys():
        key = (ticker, m)
        if key not in _ensemble_cache:
            try:
                pkg = load_model_package(m, ticker)
                _ensemble_cache[key] = pkg
            except Exception:
                pass
                
        if key in _ensemble_cache:
            loaded_models[m] = _ensemble_cache[key]
            
    if not loaded_models:
        raise ValueError(f"No models available to build ensemble for {ticker}")
        
    # Get configuration from any loaded model to know the target index
    first_pkg = list(loaded_models.values())[0]
    feature_cols = first_pkg['config']['features']
    target_name = first_pkg['config']['target']
    target_idx = feature_cols.index(target_name)
    
    predictions = []
    total_weight = 0
    
    for m_name, pkg in loaded_models.items():
        w = weights[m_name]
        m = pkg['model']
        pred_scaled = m.predict(X_input, verbose=0)[0][0]
        predictions.append(pred_scaled * w)
        total_weight += w
        
    final_scaled = sum(predictions) / total_weight
    
    # Inverse transform
    dummy = np.zeros((1, len(feature_cols)))
    dummy[0, target_idx] = final_scaled
    final_price = scaler.inverse_transform(dummy)[0, target_idx]
    
    return float(final_price)

if __name__ == "__main__":
    # Test
    try:
        pkg = load_model_package("lstm")
        print("Successfully loaded LSTM package.")
        print("Config:", pkg['config'])
    except Exception as e:
        print(f"Error loading package: {e}")
