import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
from sklearn.model_selection import train_test_split

DATA_PATH = Path("dataset/crypto_data.csv")

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the long-format crypto dataset."""
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)

def filter_data(df: pd.DataFrame, symbols: list, start_date, end_date) -> pd.DataFrame:
    """Filter dataset by selected symbols and date range."""
    return df[
        (df['symbol'].isin(symbols)) &
        (df['timestamp'] >= pd.to_datetime(start_date)) &
        (df['timestamp'] <= pd.to_datetime(end_date))
    ].copy()

def get_available_symbols(df: pd.DataFrame) -> list:
    """Get a sorted list of unique crypto symbols."""
    return sorted(df['symbol'].unique())

def get_date_range(df: pd.DataFrame):
    """Get min and max dates from the dataset."""
    return df['timestamp'].min(), df['timestamp'].max()

def get_date_slider_range(df: pd.DataFrame):
    """Return the min and max date range as a tuple for Streamlit slider."""
    min_date, max_date = get_date_range(df)
    return min_date.date(), max_date.date()

def normalize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Normalize the selected features using min-max scaling."""
    df_norm = df.copy()
    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)
    return df_norm

@st.cache_data
def reshape_lstm_data(df, feature_cols, target_col, lookback, symbols=None):
    """Reshape long-format data into LSTM-compatible sequences (symbol-wise)."""
    X_seq, y_seq = [], []

    if symbols is None:
        symbols = df["symbol"].unique()

    for symbol in symbols:
        if symbol not in df["symbol"].unique():
            continue 
        df_sym = df[df["symbol"] == symbol].sort_values("timestamp")
        X_vals = df_sym[feature_cols].values
        y_vals = df_sym[target_col].values

        for i in range(lookback, len(df_sym)):
            X_seq.append(X_vals[i - lookback:i])
            y_seq.append(y_vals[i])

    return np.array(X_seq), np.array(y_seq)

def prepare_model_data(df, model_type, test_size=0.2):
    df = df.copy()
    
    if model_type == "LSTM":
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        split_idx = int((1 - test_size) * len(df))
        df_train, df_test = df.iloc[:split_idx], df.iloc[split_idx:]
        indicator_cols = [col for col in df.columns if col not in ['timestamp', 'symbol', 'target']]
        return indicator_cols, df_train, df_test, split_idx, None, None, None, None, None
    
    else:
        symbol_col = df["symbol"].values 

        df_encoded = pd.get_dummies(df, columns=["symbol"], drop_first=True)
        dropped_col = [col for col in pd.get_dummies(df["symbol"]).columns if col not in df_encoded.columns][0] # Putting in a burner column to ensure shape matches
        df_encoded[f'symbol_{dropped_col}'] = 0
        indicator_cols = [col for col in df_encoded.columns if col not in ['timestamp', 'target']]
        
        # Ensure all numeric and drop problematic rows
        df_encoded[indicator_cols] = df_encoded[indicator_cols].apply(pd.to_numeric, errors="coerce")
        # df_encoded.dropna(subset=indicator_cols, inplace=True)

        X = df_encoded[indicator_cols].values.astype(np.float32) 
        y = df_encoded["target"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        symbols_test = symbol_col[-len(X_test):]
        
        return indicator_cols, None, None, None, X_train, X_test, y_train, y_test, symbols_test