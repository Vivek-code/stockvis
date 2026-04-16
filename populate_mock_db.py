import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
import random
import numpy as np
import time

DB_NAME = 'stock_app.db'
TICKERS = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS', 'LT.NS']
MODELS = ['lstm', 'gru', 'cnn', 'transformer', 'ENSEMBLE']

def simulate_data():
    print("Fetching historical data for 90 days back...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=95)  # Buffer 5 days 
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Ensure table exists
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

    for ticker in TICKERS:
        print(f"Loading {ticker}...")
        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        if df.empty:
            continue
        
        # Iterating through rows
        for index, row in df.iterrows():
            actual_price = float(row['Close'].iloc[0]) if hasattr(row['Close'], 'iloc') else float(row['Close'])
            date_str = index.strftime('%Y-%m-%d')
            
            # For each model, generate a "prediction" that was simulating they guessed this price 1 day before
            for model in MODELS:
                # Add synthetic noise 
                # Transformers usually have 1-2% error, LSTM/GRU 2-4%, CNN 3-5%, Ensemble ~1%
                if model == 'transformer': std = 0.015
                elif model == 'ENSEMBLE': std = 0.01
                elif model in ['lstm', 'gru']: std = 0.03
                else: std = 0.04
                
                noise = np.random.normal(0, std)
                pred_price = actual_price * (1 + noise)
                
                c.execute('''
                    INSERT INTO predictions (ticker, model, predicted_price, predicted_date, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (ticker, model, pred_price, date_str, (index - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')))
                
    conn.commit()
    conn.close()
    print("Database successfully populated with 90 days of simulated forecasts!")

if __name__ == '__main__':
    simulate_data()
