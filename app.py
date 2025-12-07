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
# Ideally, we load these once at startup
MODELS = {}
AVAILABLE_MODELS = ['lstm', 'gru', 'cnn']

def load_all_models():
    """Lengths all available models into memory."""
    print("Loading models...")
    for m_name in AVAILABLE_MODELS:
        try:
            MODELS[m_name] = load_model_package(m_name)
            print(f"Loaded {m_name}")
        except Exception as e:
            print(f"Failed to load {m_name}: {e}")

# Load models in a separate thread to not block startup if it takes long, 
# though for simple apps blocking is safer to ensure readiness.
load_all_models()

# --- 2. Routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/models')
def models():
    return render_template('models.html') # Placeholder

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

    if model_name not in MODELS:
        return jsonify({"error": f"Model {model_name} not available"}), 400

    pkg = MODELS[model_name]
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
    
    df = download_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if df is None:
        return jsonify({"error": "Failed to fetch data"}), 500
        
    # Return Close prices and Dates
    # Tail limit
    df = df.tail(limit)
    
    result = {
        "dates": df.index.strftime('%Y-%m-%d').tolist(),
        "prices": df['Close'].tolist(),
        "volumes": df['Volume'].tolist() if 'Volume' in df.columns else []
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False) # use_reloader=False to avoid loading models twice

