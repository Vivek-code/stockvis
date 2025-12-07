from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import threading

# Project imports
from data_loader import download_data
from features import add_technical_indicators
from load_model_pkg import load_model_package

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev_key_for_project_123'

# --- 1. Global State for Models ---
# Lazy Loading Cache
# Key: (ticker, model_name), Value: package
MODELS = {}
AVAILABLE_MODELS = ['lstm', 'gru', 'cnn']

def get_model(ticker, model_name):
    """
    Retrieves a model from cache or loads it from disk.
    """
    key = (ticker, model_name)
    if key in MODELS:
        return MODELS[key]
    
    print(f"Lazy loading model for {ticker} ({model_name})...")
    try:
        pkg = load_model_package(model_name, ticker)
        MODELS[key] = pkg
        return pkg
    except Exception as e:
        print(f"Failed to load {model_name} for {ticker}: {e}")
        return None

# No load_all_models() call at startup anymore for speed and memory efficiency.

# --- 2. Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/comparison')
def comparison():
    # Dynamically load all generated comparison images
    img_dir = os.path.join(app.root_path, 'static', 'images')
    images = []
    if os.path.exists(img_dir):
        for f in os.listdir(img_dir):
            if f.startswith('comparison_') and f.endswith('.png'):
                # Extract ticker logic if needed, or just use filename
                # format: comparison_TICKER.png
                ticker = f.replace('comparison_', '').replace('.png', '')
                images.append({
                    'ticker': ticker,
                    'title': ticker, # simplified
                    'file': f
                })
    return render_template('comparison.html', images=images)

# --- 3. API Endpoints ---

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Expects JSON: { "ticker": "RELIANCE.NS", "model": "lstm", "days": 1 }
    """
    data = request.json
    ticker = data.get('ticker', 'RELIANCE.NS')
    model_name = data.get('model', 'lstm').lower()
    days_to_predict = int(data.get('days', 1))

    if model_name not in AVAILABLE_MODELS:
        return jsonify({"error": f"Model type {model_name} not supported"}), 400

    pkg = get_model(ticker, model_name)
    if pkg is None:
        return jsonify({"error": f"Model {model_name} for {ticker} not found (maybe not trained yet?)"}), 404

    model = pkg['model']
    scaler = pkg['scaler']
    config = pkg['config']
    lookback = config['lookback']
    feature_cols = config['features']

    # 1. Fetch recent data
    # We need enough data to calculate indicators + lookback window
    # Safe buffer: lookback + 100 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200) 
    
    df = download_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if df is None or len(df) < lookback:
         return jsonify({"error": "Not enough recent data fetched"}), 500

    # 2. Tech Indicators
    df = add_technical_indicators(df)
    
    # 3. Prepare Input
    # We need the last 'lookback' rows
    recent_df = df.iloc[-lookback:][feature_cols]
    
    # Scale
    recent_scaled = scaler.transform(recent_df)
    
    # Reshape (1, lookback, num_features)
    input_seq = recent_scaled.reshape(1, lookback, len(feature_cols))
    
    # 4. Predict
    predictions = []
    current_seq = input_seq.copy()
    
    # Target column index (Close) for inverse scaling
    target_idx = feature_cols.index(config['target'])
    
    # Dummy array for inverse transform
    dummy = np.zeros((1, len(feature_cols)))
    
    for _ in range(days_to_predict):
        pred_scaled = model.predict(current_seq, verbose=0)
        
        # Inverse Scale the Prediction
        val = pred_scaled[0][0]
        dummy[0, target_idx] = val
        pred_price = scaler.inverse_transform(dummy)[0, target_idx]
        predictions.append(float(pred_price))
        
        # Update Sequence for next prediction (Recursive)
        # Shift everything left
        current_seq[0, :-1, :] = current_seq[0, 1:, :]
        # Update last timestep (Naive approach: assume other features stay same or use predicted price?
        # For simplicity in this project: we only update the target feature 'Close' with predicted,
        # and maybe keep others same as previous day or zero them out if unknown.
        # A better approach requires forecasting features too, but that's complex.
        # Let's clone the last step and update just the Close price.
        new_step = current_seq[0, -2, :].copy() # Copy inputs from previous "last" step 
        # Ideally we should use the new prediction. 
        # But wait, 'new_step' needs to be scaled. 'val' is already scaled.
        new_step[target_idx] = val
        current_seq[0, -1, :] = new_step

    return jsonify({
        "ticker": ticker,
        "model": model_name,
        "predictions": predictions,
        "last_close": float(df['Close'].iloc[-1]),
        "last_date": str(df.index[-1].date())
    })

@app.route('/api/history', methods=['GET'])
def history():
    ticker = request.args.get('ticker', 'RELIANCE.NS')
    limit = int(request.args.get('limit', 100))
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=limit*2) # Buffer for weekends
    
    print(f"DEBUG: Fetching history for {ticker}, limit={limit}")
    try:
        df = download_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if df is None:
            print("DEBUG: download_data returned None")
            return jsonify({"error": "Failed to fetch data (None returned)"}), 500
            
        if df.empty:
            print("DEBUG: DataFrame is empty")
            return jsonify({"error": "No data found for ticker"}), 500

        print(f"DEBUG: Data fetched. Shape: {df.shape}. Columns: {df.columns}")
        
        # Ensure 'Close' exists
        if 'Close' not in df.columns:
            print(f"DEBUG: 'Close' column missing. Available: {df.columns}")
            return jsonify({"error": f"'Close' price not found in data. columns: {df.columns}"}), 500

        # Tail limit
        df = df.tail(limit)
        
        # Handle datetime index formatting
        dates = df.index.strftime('%Y-%m-%d').tolist()
        prices = df['Close'].tolist()
        
        # Check for NaNs
        if pd.Series(prices).isna().any():
            print("DEBUG: Found NaNs in prices, filling...")
            prices = pd.Series(prices).fillna(method='ffill').tolist()

        result = {
            "dates": dates,
            "prices": prices,
            "volumes": df['Volume'].tolist() if 'Volume' in df.columns else []
        }
        print("DEBUG: Sending successful response.")
        return jsonify(result)

    except Exception as e:
        print(f"DEBUG: Exception in history endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False) # use_reloader=False to avoid loading models twice

