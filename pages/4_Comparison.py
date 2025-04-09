import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("📈 Compare Model Performance")

if "model_log" not in st.session_state or not st.session_state["model_log"]:
    st.warning("No models trained yet. Please train models on the Modeling page.")
    st.stop()

model_log = st.session_state["model_log"]

# Let user select which models to compare
model_options = [f"{i+1}. {log['model']}" for i, log in enumerate(model_log)]
selected = st.multiselect("Select models to compare:", model_options, default=model_options)

if not selected:
    st.info("Select at least one model to view comparison.")
    st.stop()

# Extract selected models from session
selected_indices = [int(s.split(".")[0]) - 1 for s in selected]
selected_logs = [model_log[i] for i in selected_indices]

# Pick metric to sort
metric_choice = st.selectbox("Sort models by:", [
    "R² (Out-of-sample)", "MAE (Out-of-sample)", "MSE (Out-of-sample)",
    "R² (In-sample)", "MAE (In-sample)", "MSE (In-sample)"
])

# Summary table
summary_data = []
for i, log in zip(selected_indices, selected_logs):
    test_metrics = {k + " (Out-of-sample)": v for k, v in log["metrics_df"].loc["Test"].to_dict().items()}
    train_metrics = {k + " (In-sample)": v for k, v in log["metrics_df"].loc["Train"].to_dict().items()}
    summary_data.append({
        "Model": f"{log['model']} #{i+1}",
        **test_metrics,
        **train_metrics
    })

ascending = not metric_choice.startswith("R²")
summary_df = pd.DataFrame(summary_data).sort_values(by=metric_choice, ascending=ascending)

st.subheader("Performance Summary")
st.dataframe(summary_df, use_container_width=True)

# Download comparison table
csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Comparison Table", csv, "model_comparison.csv", "text/csv")

# Bar chart of metric
st.subheader(f"{metric_choice} Comparison")
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(summary_df["Model"], summary_df[metric_choice], color='skyblue')
ax1.set_ylabel(metric_choice)
ax1.set_title(f"{metric_choice} by Model")
st.pyplot(fig1)

# Line chart overlay of predictions vs actual
st.subheader("Prediction vs Actual Overlay")
fig2 = go.Figure()

# Actual line
fig2.add_trace(go.Scatter(
    y=selected_logs[0]["y_test"],
    mode='lines',
    name='Actual',
    line=dict(color='black', width=2)
))

# Add predictions from all models
for i, log in enumerate(selected_logs):
    fig2.add_trace(go.Scatter(
        y=log["y_pred"],
        mode='lines',
        name=f"{log['model']} #{selected_indices[i]+1}",
        line=dict(dash='dash')
    ))

fig2.update_layout(
    height=500,
    xaxis_title="Timestep",
    yaxis_title="Value",
    showlegend=True,
    legend=dict(orientation="h", yanchor="top", y=-0.25, x=0.5, xanchor="center"),
    title="Interactive Prediction vs Actual"
)
st.plotly_chart(fig2, use_container_width=True)

# Allow user to download predictions for all models
# max_len = max(len(log["y_test"]) for log in selected_logs)
# actual_base = selected_logs[0]["y_test"]
# actual_padded = np.array(actual_base + [np.nan] * (max_len - len(actual_base)))
# pred_df = pd.DataFrame({"Actual": actual_padded})

# # Add each model's predictions, padded to match max_len
# for i, log in enumerate(selected_logs):
#     model_name = f"{log['model']} #{selected_indices[i]+1}"
#     y_pred = log["y_pred"]
#     padded_pred = np.array(y_pred + [np.nan] * (max_len - len(y_pred)))
#     pred_df[model_name] = padded_pred

# pred_df = pd.DataFrame()
# pred_csv = pred_df.to_csv(index=False).encode("utf-8")
# st.download_button("*Not working yet!* 📥 Download Predictions CSV", pred_csv, "all_model_predictions.csv", "text/csv")