import streamlit as st
import numpy as np
import pandas as pd
from utils.data_loader import load_data, get_available_symbols, prepare_model_data
from utils.models import train_model
from utils.metrics import calculate_metrics, plot_predictions, plot_residuals, calculate_train_test_metrics
import time
from tensorflow.keras.callbacks import LambdaCallback

st.set_page_config(page_title="Modeling", layout="wide")
st.title("🧠 Model Training & Evaluation")

# Load filtered data from session
if "filtered_df" not in st.session_state:
    st.warning("Please select a crypto and time range in the Data page first.")
    st.stop()

df = st.session_state["filtered_df"]

# Sidebar model selection
with st.sidebar:
    st.header("Model Configuration")
    model_choice = st.selectbox("Select a model:", [
        "Linear Regression", "Lasso Regression", "Ridge Regression", 
        "Elastic Net", 
        "PCR", "PLS", "Random Forest", "XGBoost", "CatBoost",
        "Feedforward NN", "LSTM"
    ])

    test_size = st.slider("Test set ratio:", 0.1, 0.5, 0.2, step=0.01)

    st.subheader("Hyperparameters")
    params = {}

    if model_choice == "Lasso Regression":
        params["alpha"] = st.number_input("Alpha", min_value=0.00001, max_value=10.0, value=0.001, step=0.001, format="%.3f")
    elif model_choice == "Ridge Regression":
        params["alpha"] = st.number_input("Alpha", min_value=0.0001, max_value=10.0, value=0.01, step=0.001, format="%.3f")
    elif model_choice == "Elastic Net":
        params["alpha"] = st.number_input("Alpha", min_value=0.0001, max_value=10.0, value=0.001, step=0.001, format="%.3f")
        params["l1_ratio"] = st.slider("L1 Ratio", 0.0, 1.0, 0.05, step=0.01)
    elif model_choice == "Random Forest":
        st.markdown(
                """
                <span title="⚠️ Use default parameters to avoid long wait times. ⚠️">
                    ℹ️ Hover for more info
                </span>
                """,
                unsafe_allow_html=True
            )
        st.markdown("")
        params["n_estimators"] = st.slider("n_estimators", 10, 500, 200)
        params["max_depth"] = st.slider("max_depth", 1, 20, 5)
    elif model_choice == "XGBoost":
        params["n_estimators"] = st.slider("n_estimators", 10, 500, 200)
        params["learning_rate"] = st.slider("learning_rate", 0.01, 0.5, 0.1)
        params["max_depth"] = st.slider("max_depth", 1, 10, 3)
    elif model_choice == "CatBoost":
        params["iterations"] = st.slider("Iterations", 50, 500, 200, step=50)
        params["learning_rate"] = st.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)
        params["depth"] = st.slider("Depth", 1, 6, 6)

    elif model_choice == "Feedforward NN":
        st.markdown(
                """
                <span title="⚠️ Use default parameters to avoid long wait times. ⚠️">
                    ℹ️ Hover for more info
                </span>
                """,
                unsafe_allow_html=True
            )
        st.markdown("")
        params["epochs"] = st.slider("Epochs", 10, 200, 200)
    elif model_choice == "LSTM":
        st.markdown(
                """
                <span title="⚠️ Use default parameters to avoid long wait times. ⚠️">
                    ℹ️ Hover for more info
                </span>
                """,
                unsafe_allow_html=True
            )
        st.markdown("")
        params["epochs"] = st.slider("Epochs", 10, 200, 200)
        params["lookback"] = st.slider("Lookback steps", 1, 20, 5)

    train_button = st.button("Train Model")

indicator_cols, df_train, df_test, split_idx, X_train, X_test, y_train, y_test, symbols_test = prepare_model_data(df, model_choice, test_size)
symbols = get_available_symbols(df)

if not train_button:
    st.info("👈 Please select a model and click **Train Model** using the sidebar to begin.")
    if model_choice == "LSTM":
        st.markdown("❗LSTM may take more than 30 minutes to train depending on if default model is used and your dataset.❗")
        st.markdown("""
        ### Available Pre-trained models:
        1. Epochs: 200, Lookback: 5
        2. Epochs: 100, Lookback: 5""")
    elif model_choice == "Random Forest":
        st.markdown("""
        ### Available Pre-trained models:
        1. n_estimators: 200, max_depth: 5""")
    elif model_choice == "Feedforward NN":
        st.markdown("""
        ### Available Pre-trained models:
        1. Epochs: 200
        2. Epochs: 100""")
    st.stop()

# Training
if train_button:
    with st.spinner("Training model..."):
        if model_choice == "LSTM":
            st.info("LSTM uses a single 64 neuron layer with ReLU activation function as a simple architecture avoids overfitting.")
            
            target_col = "target"
            model, y_pred, y_test_lstm, X_train_lstm, y_train_lstm, X_test_lstm = train_model(
                model_choice,
                None, None, None,  # Unused for LSTM
                params,
                df_train=df_train,
                df_test=df_test,
                symbols=symbols,
                feature_cols=indicator_cols,
                target_col=target_col)

            with st.spinner("Predicting..."):
                metrics_df, train_pred, test_pred = calculate_train_test_metrics(model, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm)
            
            st.success("Training complete!")
            st.subheader("Performance Metrics")
            st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

            st.subheader("Prediction vs Actual")
            st.pyplot(plot_predictions(y_test_lstm, y_pred))

            st.subheader("Residuals")
            st.pyplot(plot_residuals(y_test_lstm, y_pred))

            y_test = y_test_lstm.tolist()
            y_pred = y_pred.tolist()


        else: # Any model other than LSTM
            if model_choice == "Linear Regression":
                st.info("💡 This is the standard Ordinary Least Squares: ")
                st.latex(r"y_i = \mathbf{x}_i^\top \boldsymbol{\beta}")
            elif model_choice == "Lasso Regression":
                st.info("Lasso performs regression with L1 penalty to encourage sparsity.")
                st.latex(r"\hat{\boldsymbol{\beta}} = \arg\min_\beta \|X\beta - Y\|_2^2 + \lambda \|\beta\|_1")
            elif model_choice == "Ridge Regression":
                st.info("Ridge adds an L2 penalty to shrink coefficients.")
                st.latex(r"\hat{\boldsymbol{\beta}} = \arg\min_\beta \|X\beta - Y\|_2^2 + \lambda \|\beta\|_2^2")
            elif model_choice == "PCR":
                st.info("Principal Component Regression projects features into principal components before regression.")
                st.latex(r"y_i = \mathbf{x}_i^\top \mathbf{P}")
            elif model_choice == "PLS":
                st.info("Partial Least Squares uses components from PLS weights.")
                st.latex(r"y_i = \mathbf{x}_i^\top \mathbf{W}")
            elif model_choice == "Random Forest":
                st.info("Random Forest averages the predictions from multiple decision trees.")
                st.latex(r"\hat{y} = \frac{1}{N} \sum_{i=1}^N f_i(x)")
            elif model_choice == "XGBoost":
                st.info("XGBoost builds trees builds on Random Forests by incorporating Regularization and Optimization.")
            elif model_choice == "CatBoost":
                st.info("CatBoost uses a permulation driven approach to build trees and excels at categorical data.")
            elif model_choice == "Feedforward NN":
                st.info("FFNN uses 3 hidden layers (32, 16, 8 neurons respectively) with ReLU activation to prevent vanishing gradients.")

            model, y_pred = train_model(model_choice, X_train, y_train, X_test, params)

            with st.spinner("Predicting..."):
                metrics_df, train_pred, test_pred = calculate_train_test_metrics(model, X_train, y_train, X_test, y_test)

            st.success("Training complete!")
            st.subheader("Performance Metrics")
            st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

            st.subheader("Prediction vs Actual")
            st.pyplot(plot_predictions(y_test, y_pred))

            st.subheader("Residuals")
            st.pyplot(plot_residuals(y_test, y_pred))

        # Save model + results to session state
        log_entry = {
            "model": model_choice,
            "params": params,
            "metrics_df": metrics_df,
            "y_test": y_test,
            "y_pred": y_pred
        }

        if "model_log" not in st.session_state:
            st.session_state["model_log"] = []

        st.session_state["model_log"].append(log_entry)

        # Download prediction as CSV
        if model_choice == "LSTM":
            num_rows = len(y_test_lstm)
            symbol_seq = []
            for symbol in symbols:
                count = len(df_test[df_test["symbol"] == symbol]) - params["lookback"]
                symbol_seq.extend([symbol] * max(count, 0))

            download_df = pd.DataFrame({"Symbol": symbol_seq[:num_rows],
            "Actual": y_test_lstm, 
            "Predicted": y_pred})
        else:
            download_df = pd.DataFrame({
                "Symbol": symbols_test,
                "Actual": y_test,
                "Predicted": y_pred
            })
        csv = download_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")