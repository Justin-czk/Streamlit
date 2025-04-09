import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from pathlib import Path

# Settings
NUM_TIMESTEPS = 1000
CRYPTO_SYMBOLS = ["BTC", "ETH", "XRP"]
NUM_INDICATORS = 50
OUTPUT_PATH = Path("dataset/crypto_data.csv")

def generate_indicator_data(rows, num_features):
    return np.random.randn(rows, num_features) * np.random.uniform(0.5, 2.0, size=(1, num_features))

def create_crypto_data(symbol, start_date):
    dates = [start_date + timedelta(days=i) for i in range(NUM_TIMESTEPS)]
    indicators = generate_indicator_data(NUM_TIMESTEPS, NUM_INDICATORS)
    
    df = pd.DataFrame(indicators, columns=[f"indicator_{i+1}" for i in range(NUM_INDICATORS)])
    df["timestamp"] = dates
    df["symbol"] = symbol

    # Create a synthetic target using a few indicators
    df["target"] = (
        0.2 * df["indicator_1"] -
        0.4 * df["indicator_5"] +
        0.3 * df["indicator_10"] +
        np.random.normal(0, 0.5, NUM_TIMESTEPS)
    )
    
    return df

def main():
    start_date = datetime(2020, 1, 1)
    all_data = pd.concat([create_crypto_data(symbol, start_date) for symbol in CRYPTO_SYMBOLS])
    all_data.sort_values(by=["symbol", "timestamp"], inplace=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_data.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Mock dataset saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
