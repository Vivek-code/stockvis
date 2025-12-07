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
            # Flatten columns: if we have (Price, Ticker), we just want Price.
            # Usually level 0 is Price (Close, Open, etc) and level 1 is Ticker.
            # If we only have 1 ticker, we can drop the ticker level.
            try:
                # Assuming level 1 is the ticker, we drop it.
                if df.columns.nlevels > 1:
                    df.columns = df.columns.droplevel(1)
            except Exception as e:
                print(f"Error flattening columns: {e}")
        
        # Verify columns are flat now
        print(f"Columns after download: {df.columns}")

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
