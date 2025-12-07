import pandas as pd
import pandas_ta as ta

def add_technical_indicators(df):
    """
    Adds technical indicators to the dataframe.
    Expects columns: Open, High, Low, Close, Volume.
    """
    # Create a copy to avoid SettingWithCopy warnings on the original
    df = df.copy()

    # Ensure columns are properly named (yfinance might give 'Adj Close' etc)
    # yfinance typically gives: Open, High, Low, Close, Adj Close, Volume
    # If using MultiIndex, we might need to flatten or access specific level.
    # Assuming df is single-index columns for now. 
    
    # Check if columns are MultiIndex (common with recent yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        # We assume the top level is Price and columns are properly aligned
        # But pandas_ta expects flat columns 'Open', 'High', etc.
        try:
            # Flatten or extract if needed. Simple fix for recent yfinance:
            # If columns are ('Close', 'RELIANCE.NS'), we want just 'Close'.
            # If it's just 'Close', fine.
            if df.columns.nlevels > 1:
                df.columns = df.columns.droplevel(1) # Drop Ticker level
        except:
            pass

    # Rename to Title Case just in case (pandas_ta prefers them)
    # yfinance is usually Title case already: Open, High, Low, Close, Volume.

    # 1. RSI (14)
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # 2. MACD (12, 26, 9)
    # macd method returns a generic DF with columns like MACD_12_26_9, MACDh_..., MACDs_...
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
        # Rename for easier access
        df.rename(columns={
            'MACD_12_26_9': 'MACD',
            'MACDh_12_26_9': 'MACD_hist',
            'MACDs_12_26_9': 'MACD_signal'
        }, inplace=True)

    # 3. Bollinger Bands (20, 2)
    bb = ta.bbands(df['Close'], length=20, std=2)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)
        df.rename(columns={
            'BBU_20_2.0': 'BB_upper',
            'BBM_20_2.0': 'BB_mid',
            'BBL_20_2.0': 'BB_lower'
        }, inplace=True)

    # 4. ATR (14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # 5. ROC (Rate of Change)
    df['ROC'] = ta.roc(df['Close'], length=10)

    # 6. SMA (10, 20)
    df['SMA_10'] = ta.sma(df['Close'], length=10)
    df['SMA_20'] = ta.sma(df['Close'], length=20)

    # Drop NaNs created by indicators (e.g. first 26 rows for MACD)
    df.dropna(inplace=True)

    return df

if __name__ == "__main__":
    # Test
    from data_loader import download_data
    df = download_data("RELIANCE.NS", "2020-01-01", "2021-01-01")
    if df is not None:
        df_new = add_technical_indicators(df)
        print("Columns with indicators:", df_new.columns)
        print(df_new.head())
