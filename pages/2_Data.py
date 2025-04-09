import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from utils.data_loader import load_data, get_available_symbols, get_date_slider_range, filter_data
from sklearn.decomposition import PCA

st.set_page_config(page_title="Data Explorer", layout="wide")

st.title("📊 Data Exploration")

# Load full dataset
df = load_data()

# Sidebar filters
with st.sidebar:
    st.header("Filter Data")
    symbol_option = st.radio( # Default to "All" symbols to get most accurate results
        "Which symbols to include?",
        ["All", "Select specific"],
        index=0
        )

    symbols = get_available_symbols(df)
    if symbol_option == "All":
        selected_symbols = symbols
    else:
        selected_symbols = st.multiselect("Select crypto symbols:", symbols, default=symbols[:3])

    # Date range slider
    min_date, max_date = get_date_slider_range(df)
    selected_date_range = st.slider("Select date range:",
                                    min_value=min_date,
                                    max_value=max_date,
                                    value=(min_date, max_date),
                                    format="YYYY-MM-DD")

st.info("""👈 You may filter for some Crypto Symbols using the sidebar. 

**Note:** Model Performance may get worse with less data.""")

# General dataset information
st.markdown("""
# Dataset Preprocessing Information
## Data Sources
- Crypto data is sourced from Coindesk through the Cryptocompare API and Etherscan.
- Macroeconomic data are sourced from FRED, Prognosis API, Yahoo Finance API.
- Sentiment data are sourced from Reddit, FRED, Google’s GDELT News database.

## Preprocessing Steps
1. Data is aggregated to daily frequency.
2. Stationarity: Non-stationary features (ADF test) were appropriately differenced.
3. Missing Data: Gaps in data were imputed using the following methods:
    
    3a. Continuous numerical data: Linear interpolation to ensure smooth transitioning.
    
    3b. Non-continuous numerical data: Forward fill to ensure logical sense. Eg., Fed Funds Rate, since the community meets up only 8 times a year to set target rate. Missing values should be brought forward.
    
    3c. Sentiment data: Imputed with 0, representing neutral sentiment for time periods with no data points.
4. Clipping: Target Variable (returns) clipped to control returns outside the middle 98% of data. This clipped 8 data-points. Clipping done to ensure large outliers do not distort prediction magnitudes. 
5. Scaling: Features standardization using StandardScaler to ensure they are comparable.
6. Lag: All features are lagged to ensure that prediction is done with knowledge available before returns are realized.
""")

# Filter data based on user selection
filtered_df = filter_data(df, selected_symbols, selected_date_range[0], selected_date_range[1])
filtered_df["timestamp"] = pd.to_datetime(filtered_df["timestamp"])
st.session_state["filtered_df"] = filtered_df

st.subheader(f"Filtered Dataset: {len(filtered_df):,} rows")
st.dataframe(filtered_df.head(), use_container_width=True)

# Summary statistics (Excluded counts, means and SD due to pre-processing)
st.subheader("Summary Statistics")
st.write(filtered_df.drop(columns=["timestamp"]).describe().loc[["min", "25%", "50%", "75%", "max"]])

# Correlation heatmap 
st.subheader("Correlation Matrix")
indicator_cols = [col for col in filtered_df.columns if col not in ['timestamp', 'symbol', 'target']]
corr = filtered_df[indicator_cols].corr()

fig1, ax1 = plt.subplots(figsize=(6, 6))
sns.heatmap(corr, ax=ax1, cmap='coolwarm', annot=False, linewidths=0.2)
st.pyplot(fig1)

# Line chart of indicator values
st.subheader("Line Chart of Selected Indicators")
selected_indicators = st.multiselect("Choose indicators to plot:", indicator_cols, default=indicator_cols[:3])

# Limit symbols shown in chart to avoid clutter and complexity
if symbol_option == "All":
    chart_symbols = [s for s in ["BTC", "ETH", "DOGE"] if s in selected_symbols]
    st.caption("Showing BTC, ETH, and DOGE by default when 'All' is selected.")
else:
    chart_symbols = selected_symbols[:5]
    if len(selected_symbols) > 5:
        st.warning("Showing only the first 5 selected symbols to avoid clutter.")

for symbol in chart_symbols:
    st.markdown(f"**{symbol}**")
    chart_data = filtered_df[filtered_df['symbol'] == symbol].set_index('timestamp')[selected_indicators]
    st.line_chart(chart_data)