import os
import json
import joblib
from tensorflow.keras.models import load_model

def load_model_package(model_name, base_dir="models"):
    """
    Loads a model package (model, scaler, config) by name.
    
    Args:
        model_name (str): Name of the model directory (e.g., 'lstm', 'gru').
        base_dir (str): output directory of models.
        
    Returns:
        dict: {
            'model': keras_model,
            'scaler': scaler_obj,
            'config': dict
        }
    """
    model_dir = os.path.join(base_dir, model_name)
    
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
    # Try different extensions if needed, but we saved as .keras
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

if __name__ == "__main__":
    # Test
    try:
        pkg = load_model_package("lstm")
        print("Successfully loaded LSTM package.")
        print("Config:", pkg['config'])
    except Exception as e:
        print(f"Error loading package: {e}")
