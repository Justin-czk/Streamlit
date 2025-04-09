# Cryptocurrency Modeling App

This Streamlit app lets users explore and model time-series cryptocurrency data using regression, tree-based, and neural network models.

## Features
- EDA, PCA, and correlation heatmaps
- Regression, Random Forest, XGBoost, Feedforward NN, and LSTM models
- Visual evaluation and side-by-side model comparison

## Getting Started

1. Clone this repo:
```bash
git clone https://github.com/Justin-czk/Streamlit.git
cd Streamlit

2. Download dataset here:
https://drive.google.com/file/d/1sqo2Pt7LRK4cUU9sEnJtGFx9YdGNKWI4/view?usp=sharing

3. Install dependencies:
`pip install -r requirements.txt`

4. Run the app:
`streamlit run Home.py`

## Directory Structure
crypto-modeling-app/
│
├── .streamlit/
│   └── config.toml
│
├── assets/
│   ├── BMA_tests.png
│   ├── BMA_ts.png
│   └── style.css
│
├── dataset/
│   └── crypto_data.csv
│
├── pages/
│   ├── 2_Data.py
│   ├── 3_Modelling.py
│   ├── 4_Comparison.py
│   └── 5_Final Model.py
│
├── trained_models/
│   ├── FFNN_30860c006e13696fbd4dafcab8179fa0.keras
│   ├── FFNN_b1eeb53e342bdc7828c708a099ceb6e9.keras
│   ├── LSTM_770a532605f086afa8cd98be65edb172.keras
│   ├── LSTM_8662e15cd642d63553f08b7b8bca0bb5.keras
│   └── RF_e4b26126cadabc9e4ec83a253709789f.pkl
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   └── metrics.py
│
├── Home.py
├── requirements.txt
└── README.md
