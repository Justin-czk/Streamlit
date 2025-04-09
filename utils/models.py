import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.optimizers import Adam
# from groupyr import SGLCV
from catboost import CatBoostRegressor
from utils.data_loader import reshape_lstm_data #, generate_groups
import hashlib
import json
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import os
import pickle

import numpy as np
import random
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def train_model(model_name, X_train, y_train, X_test, params, **kwargs):

    if model_name == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_name == "Lasso Regression":
        model = Lasso(alpha=params["alpha"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_name == "Ridge Regression":
        model = Ridge(alpha=params["alpha"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_name == "Elastic Net":
        l1_ratio = params.get("l1_ratio", 0.5)
        model = ElasticNet(alpha=params.get("alpha", 1.0),
                           l1_ratio=l1_ratio,
                           random_state=SEED)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_name == "PCR":
        pca = PCA()
        linreg = LinearRegression()
        model = Pipeline([("pca", pca), ("linreg", linreg)])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_name == "PLS":
        model = PLSRegression(n_components=min(5, X_train.shape[1]))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test).ravel()

    elif model_name == "Random Forest":

        model_dir = Path("trained_models")
        model_dir.mkdir(exist_ok=True)
        model_hash = get_model_hash(model_name, params)
        model_path = model_dir / f"RF_{model_hash}.pkl"

        if model_path.exists():
            print("🔁 Loading pre-trained model from disk... 🔁")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            y_pred = model.predict(X_test) 
            return model, y_pred

        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=SEED
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    elif model_name == "XGBoost":
        model = XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=SEED
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    elif model_name == "CatBoost":
        iterations = params.get("iterations", 200)
        learning_rate = params.get("learning_rate", 0.1)
        depth = params.get("depth", 6)

        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            verbose=0,
            random_seed=SEED
            )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_name == "Feedforward NN":

        # print("Training Feedforward NN...") 
        model_dir = Path("trained_models")
        model_dir.mkdir(exist_ok=True)
        model_hash = get_model_hash(model_name, params)
        model_path = model_dir / f"FFNN_{model_hash}.keras"
        print(model_path)
        if model_path.exists():
            print("🔁 Loading pre-trained model from disk... 🔁")
            model = load_model(model_path)
            y_pred = model.predict(X_test).ravel()
            return model, y_pred

        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(), loss='mse')
        model.fit(X_train, y_train, epochs=params["epochs"], verbose=0)
        y_pred = model.predict(X_test).ravel()

        model.save(model_path)  # Save model to /trained_models

    elif model_name == "LSTM":
        df_train = kwargs.get("df_train")
        df_test = kwargs.get("df_test")
        symbols = kwargs.get("symbols")
        feature_cols = kwargs.get("feature_cols")
        target_col = kwargs.get("target_col")
        symbols = kwargs.get("symbols")
        lookback = params["lookback"]
        # callback = kwargs.get("progress_callback", None)
        # progress_callback = [callback] if callback is not None else []
        # callbacks = [cb for cb in [kwargs.get("progress_callback", None)] if cb is not None]

        X_train_lstm, y_train_lstm = reshape_lstm_data(df_train, feature_cols, target_col, lookback, symbols=symbols)
        X_test_lstm, y_test_lstm = reshape_lstm_data(df_test, feature_cols, target_col, lookback, symbols=symbols)

        # Check if the model has already been trained and saved
        model_dir = Path("trained_models")
        model_dir.mkdir(exist_ok=True)
        model_hash = get_model_hash(model_name, params, extra={ 
            "symbols": symbols, 
            "lookback": lookback, 
            "features": feature_cols 
        })
        model_path = model_dir / f"LSTM_{model_hash}.keras"

        if model_path.exists():
            print("🔁 Loading pre-trained model from disk... 🔁")
            model = load_model(model_path)
            y_pred = model.predict(X_test_lstm).ravel()
            return model, y_pred, y_test_lstm, X_train_lstm, y_train_lstm, X_test_lstm

        model = Sequential()
        model.add(LSTM(64, input_shape=(lookback, len(feature_cols))))
        model.add(Dense(1))
        model.compile(optimizer=Adam(), loss=MeanSquaredError())
        model.fit(X_train_lstm, y_train_lstm, epochs=params["epochs"], verbose=0) #progress_callback, callbacks=callbacks

        model.save(model_path, save_format="keras") 

        y_pred = model.predict(X_test_lstm).ravel()

        return model, y_pred, y_test_lstm, X_train_lstm, y_train_lstm, X_test_lstm

    else:

        raise ValueError(f"Unsupported model: {model_name}")

    return model, y_pred

def get_model_hash(model_name, params, extra=None):
    hash_input = {
        "model": model_name,
        "params": params,
        "extra": extra 
    }
    hash_str = json.dumps(hash_input, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()