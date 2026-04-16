from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import sqlite3
import csv
import io
import threading
import time
import yfinance as yf

# Project imports
from data_loader import download_data
from features import add_technical_indicators
from load_model_pkg import load_model_package, ensemble_predict
from sentiment_analyzer import get_sentiment_score

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev_key_for_project_123'

# --- 1. Global State for Models ---
# Lazy Loading Cache
# Key: (ticker, model_name), Value: package
MODELS = {}
AVAILABLE_MODELS = ['lstm', 'gru', 'cnn', 'transformer']

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


# --- DB Setup ---
DB_NAME = 'stock_app.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            model TEXT NOT NULL,
            predicted_price REAL NOT NULL,
            predicted_date TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")

# Initialize DB on startup
init_db()

def save_prediction_to_db(ticker, model, price, date_str):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('INSERT INTO predictions (ticker, model, predicted_price, predicted_date) VALUES (?, ?, ?, ?)',
                  (ticker, model, price, date_str))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

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

@app.route('/api/metrics')
def get_metrics():
    """
    Returns performance metrics for all trained models.
    """
    base_dir = "models"
    metrics_data = []
    
    if os.path.exists(base_dir):
        for ticker in os.listdir(base_dir):
            ticker_path = os.path.join(base_dir, ticker)
            if not os.path.isdir(ticker_path):
                continue
                
            for model_name in os.listdir(ticker_path):
                model_path = os.path.join(ticker_path, model_name)
                config_path = os.path.join(model_path, 'config.json')
                
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            metrics_data.append({
                                'ticker': config.get('ticker', ticker),
                                'model': config.get('model_type', model_name),
                                'mae': config.get('mae'),
                                'rmse': config.get('rmse'),
                                'r2': config.get('r2', 0), # Default to 0 if not present
                                'mape': config.get('mape', 0),
                                'date_trained': config.get('date_trained')
                            })
                    except Exception as e:
                        print(f"Error reading config for {ticker}/{model_name}: {e}")
                        
    return jsonify(metrics_data)



@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Expects JSON: { "ticker": "RELIANCE.NS", "model": "lstm", "days": 1 }
    """
    data = request.json
    ticker = data.get('ticker', 'RELIANCE.NS')
    model_name = data.get('model', 'lstm').lower()
    days_to_predict = int(data.get('days', 1))

    if model_name not in AVAILABLE_MODELS and model_name != 'ensemble':
        return jsonify({"error": f"Model type {model_name} not supported"}), 400

    if model_name == 'ensemble':
        # Need config and scaler from one of the basic models to prepare data
        pkg = get_model(ticker, 'lstm') or get_model(ticker, 'gru') or get_model(ticker, 'cnn') or get_model(ticker, 'transformer')
        if pkg is None:
            return jsonify({"error": f"No models found for {ticker} to build ensemble"}), 404
        model = None
    else:
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

    # 2b. Inject live sentiment if model expects it
    if 'sentiment' in feature_cols:
        try:
            sentiment_score, _ = get_sentiment_score(ticker)
            df['sentiment'] = 0.0  # Default for all rows
            df.iloc[-1, df.columns.get_loc('sentiment')] = sentiment_score
            print(f"Injected live sentiment: {sentiment_score:.4f}")
        except Exception as e:
            print(f"Sentiment injection failed: {e}")
            if 'sentiment' not in df.columns:
                df['sentiment'] = 0.0
    
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
        if model_name == 'ensemble':
            # ensemble_predict computes the weighted average and inverse transforms it directly
            pred_price = ensemble_predict(ticker, current_seq, scaler)
            dummy[0, target_idx] = pred_price
            val = scaler.transform(dummy)[0, target_idx] # re-scale to inject back to sequence
        else:
            pred_scaled = model.predict(current_seq, verbose=0)
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

    last_actual_date = df.index[-1].date()
    
    # Save to DB (Only the first prediction for simplicity, or all)
    # Let's save the first day prediction as the "Forecast"
    pred_date_obj = last_actual_date + timedelta(days=1)
    db_model_name = "ENSEMBLE" if model_name == "ensemble" else model_name
    save_prediction_to_db(ticker, db_model_name, predictions[0], str(pred_date_obj))

    return jsonify({
        "ticker": ticker,
        "model": model_name,
        "predictions": predictions,
        "last_close": float(df['Close'].iloc[-1]),
        "last_date": str(last_actual_date)
    })

@app.route('/api/predictions')
def get_predictions():
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM predictions ORDER BY created_at DESC LIMIT 50')
        rows = c.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                'id': row['id'],
                'ticker': row['ticker'],
                'model': row['model'],
                'price': row['predicted_price'],
                'date': row['predicted_date'],
                'created_at': row['created_at']
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/export_predictions')
def export_predictions():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('SELECT ticker, model, predicted_price, predicted_date, created_at FROM predictions ORDER BY created_at DESC')
        rows = c.fetchall()
        conn.close()
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Ticker', 'Model', 'Predicted Price', 'Target Date', 'Timestamp'])
        writer.writerows(rows)
        
        return output.getvalue(), 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=prediction_history.csv'
        }
    except Exception as e:
        return str(e), 500

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

@app.route('/api/sentiment/<ticker>')
def get_sentiment(ticker):
    """
    Returns sentiment analysis for a given ticker.
    """
    try:
        score, headlines = get_sentiment_score(ticker)
        headlines_count = len(headlines)
        
        if score > 0.1:
            label = "Positive"
        elif score < -0.1:
            label = "Negative"
        else:
            label = "Neutral"

        return jsonify({
            "ticker": ticker,
            "score": round(score, 4),
            "label": label,
            "headlines_analyzed": headlines_count,
            "headlines": headlines
        })
    except Exception as e:
        return jsonify({
            "ticker": ticker,
            "score": 0.0,
            "label": "Neutral",
            "headlines_analyzed": 0,
            "headlines": [],
            "error": str(e)
        })
# --- Reality Check Feature ---
YF_CACHE = {} # dict mapping (ticker, date_str) to price
YF_CACHE_TIMESTAMP = 0

def refresh_yf_cache_if_needed(tickers):
    global YF_CACHE_TIMESTAMP
    current_time = time.time()
    if current_time - YF_CACHE_TIMESTAMP < 3600 and YF_CACHE:
        return # Cache is fresh
    
    YF_CACHE.clear()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=95)
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
            if df.empty:
                continue
            for index, row in df.iterrows():
                date_str = index.strftime('%Y-%m-%d')
                val = row['Close']
                price = float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
                YF_CACHE[(ticker, date_str)] = price
        except Exception as e:
            print(f"Error caching {ticker}: {e}")
            
    YF_CACHE_TIMESTAMP = current_time

@app.route('/reality-check')
def reality_check():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    ninety_days_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    c.execute('''
        SELECT ticker, model, predicted_price, predicted_date 
        FROM predictions 
        WHERE predicted_date >= ?
        ORDER BY predicted_date DESC
    ''', (ninety_days_ago,))
    
    rows = c.fetchall()
    conn.close()
    
    unique_tickers = list(set([r[0] for r in rows]))
    refresh_yf_cache_if_needed(unique_tickers)
    
    processed_results = []
    
    for ticker, model, pred_price, pred_date_str in rows:
        cache_key = (ticker, pred_date_str)
        if cache_key in YF_CACHE:
            actual_price = YF_CACHE[cache_key]
            error_inr = pred_price - actual_price
            error_pct = abs(error_inr) / actual_price * 100
            
            if error_pct < 2: rating = 'Excellent'
            elif error_pct < 5: rating = 'Good'
            elif error_pct < 10: rating = 'Fair'
            else: rating = 'Poor'
            status = 'Complete'
        else:
            actual_price = None
            error_inr = None
            error_pct = None
            rating = 'Pending'
            status = 'Pending'
            
        processed_results.append({
            'date': pred_date_str,
            'ticker': ticker,
            'model': model.upper(),
            'predicted_price': round(pred_price, 2),
            'actual_price': round(actual_price, 2) if actual_price else None,
            'error_inr': round(error_inr, 2) if error_inr else None,
            'error_pct': round(error_pct, 2) if error_pct else None,
            'rating': rating,
            'status': status
        })
        
    valid_results = [r for r in processed_results if r['status'] == 'Complete']
    if valid_results:
        summary = {
            'mean_inr': round(sum([abs(r['error_inr']) for r in valid_results]) / len(valid_results), 2),
            'mean_pct': round(sum([r['error_pct'] for r in valid_results]) / len(valid_results), 2),
            'best': min(valid_results, key=lambda x: x['error_pct']),
            'worst': max(valid_results, key=lambda x: x['error_pct'])
        }
    else:
        summary = None
        
    return render_template('reality_check.html', results=processed_results, summary=summary)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False) # use_reloader=False to avoid loading models twice
