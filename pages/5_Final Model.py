import streamlit as st
import pandas as pd

st.set_page_config(page_title="Bayesian Model Averaging", layout="wide")
st.title("Bayesian Model Averaging (BMA) Summary")
st.info("Results in this page are hardcoded due to complexity in model")
st.markdown("")
st.markdown("")

weights_path = "dataset/bma_weights.csv"
results_path = "dataset/bma_results.csv"

try:
    weights_df = pd.read_csv(weights_path)
    results_df = pd.read_csv(results_path)
except FileNotFoundError:
    st.error("BMA results not found. Please ensure you've run the BMA pipeline.")
    st.stop()

col1, col2 = st.columns(2)

with col1:

    st.subheader("BMA Prediction vs Actual (Test Set)")
    st.image("assets/BMA_ts.png", use_container_width=True, caption="Time Series Plot showing decent fit but see some complexities in predicting magnitude.")
    
    st.subheader("📉 Residual Diagnostics")
    st.image("assets/BMA_tests.png", caption="Histogram & Q-Q Plot", use_container_width=True)

with col2:
    st.markdown("### 🔎 Statistical Tests")
    st.markdown("""
- **Shapiro-Wilk Test**  
  *W = 0.9356, p = 0.0000*  
  → Residuals are **not normally distributed**. This is expected in financial/time series data and doesn't invalidate the model.

- **Jarque-Bera Test**  
  *JB = 18736.86, p = 0.0000*  
  → Confirms the presence of **skewness or excess kurtosis** in residuals.

- **Ljung-Box Test (lag=10)**  
  *LB-stat = 239.19, p = 1.01e-45*  
  → Strong evidence of **autocorrelation** in residuals. Suggests possible missing temporal features.

- **Durbin-Watson Statistic**  
  *DW = 2.1315*  
  → Close to 2, which indicates **no serious lag-1 autocorrelation**.
    """)

st.subheader("📊 Model Weights in BMA Ensemble")
st.dataframe(weights_df, use_container_width=True)

# === Section 4: Model Performance Table ===
st.subheader("📋 Model Performance Summary")
st.dataframe(results_df, use_container_width=True)



st.download_button(
      "📥 Download BMA Results CSV",
      results_df.to_csv(index=False).encode("utf-8"),
      "bma_results.csv",
      "text/csv"
  )


st.download_button(
      "📥 Download BMA Weights CSV",
      weights_df.to_csv(index=False).encode("utf-8"),
      "bma_weights.csv",
      "text/csv"
  )
