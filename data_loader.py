import yfinance as yf
import pandas as pd
import os

def download_data(ticker, start_date, end_date):
    """
    Downloads historical stock data from Yahoo Finance.
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            print(f"Warning: No data found for {ticker}")
            return None
        
        # yfinance might return MultiIndex columns (Price, Ticker) if downloading multiple,
        # but for single ticker it might be simpler.
        # Let's ensure we have a clean DataFrame.
        if isinstance(df.columns, pd.MultiIndex):
             # If columns are (Price, Ticker), we might drop the ticker level if it's there
             # Or sometimes it's (Ticker, Price).
             # yfinance update: .download() often returns MultiIndex columns even for single ticker now.
             pass 

        # Forward fill generic missing values just in case
        df.ffill(inplace=True)
        return df
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return None

if __name__ == "__main__":
    # Test
    ticker = "RELIANCE.NS"
    df = download_data(ticker, "2015-01-01", "2023-01-01")
    if df is not None:
        print(f"Downloaded {len(df)} rows.")
        print(df.head())
