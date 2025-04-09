import streamlit as st
import pandas as pd
from utils.data_loader import load_data, get_date_range, get_available_symbols

st.title("🏠 Empirical Cryptocurrency Pricing using Machine Learning")

def inject_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

inject_css()

st.markdown("✅ Use the sidebar to navigate through the app. ✅")

st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center;">
        <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbXVvaHJ3bWNicG04NHRsbmNrYXF2Z2QxMnA4MnJweXNzcmd0c2lkZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/JLBCG3FymKLsqt87A6/giphy.gif" style="width: 30 %; max-width: 500px;
            border-radius: 0.5rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);" />
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("")

st.markdown("""
Welcome to my Cryptocurrency Modeling App made for **DSE4101**!

This interactive app allows you to:
- Explore cryptocurrency time-series data and indicators 
- Train various machine learning models
- Evaluate model performance and download results
- Compare models side-by-side with visual insights

---

### **Navigation Guide**
Please view them in order!
- **📊 Data**: Explore historical data, visualize trends and indicators
- **🧠 Modeling**: Train regression, tree-based, or neural network models
- **📈 Comparison**: Compare model performance across different runs

---

### **About the Data**
This app uses a pre-cleaned dataset containing:
- **Multiple cryptocurrencies** chosen to ensure activity on prominent exchanges
- **50 indicators** selected to consider a broad range of different indicators
- **1000 time periods** to ensure enough historical data for training
""")

# Show sample data preview
df = load_data()
symbols = get_available_symbols(df)
min_date, max_date = get_date_range(df)
# Prepare data for sample display
btc_sample = df[df["symbol"] == "BTC"].head(5)
priority_cols = ["symbol", "timestamp", "target", "RSI", "fear_greed_index", "FEDFUNDS", "gas_price_gwei", "title_sentiment", "ROC"]
remaining_cols = [col for col in btc_sample.columns if col not in priority_cols]
ordered_cols = priority_cols + remaining_cols
btc_sample = btc_sample[ordered_cols]

st.markdown(f"**Date range:** {min_date.date()} to {max_date.date()}")
with st.expander("Click to view all available symbols"):
    df_symbols = pd.DataFrame(symbols, columns=["Symbol"])
    df_symbols.index = df_symbols.index + 1  # Make index start at 1
    df_symbols.index.name = "No."            # Rename the index column
    st.dataframe(df_symbols, use_container_width=True)

st.markdown("**Sample of the dataset:**")
st.dataframe(btc_sample, use_container_width=True) # Show first 5 rows of BTC data

st.markdown("""
### **Disclaimer:** 
This app is for educational purposes only. The data used is not financial advice.

This Individual Project is made by **Justin Cheong** for **DSE4101** course at NUS. 

ChatGPT-4o was used for debugging.""")
st.markdown("It builds on the group project by Wan Ting, Brandon and I, which references the paper by **Shihao Gu, Bryan Kelly and Dacheng Xiu (April 2019)**, Empirical Asset Pricing via Machine Learning.")