import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Time Series Anomaly Detection", layout="wide")

# -----------------------------
# Autoencoder Model
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, seq_len)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# -----------------------------
# Sequence creation
# -----------------------------
def create_sequences(data, seq_len=30):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

# -----------------------------
# Train model (cached)
# -----------------------------
@st.cache_resource
def train_model(_X, seq_len):
    model = Autoencoder(seq_len)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        output = model(_X)
        loss = criterion(output, _X)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return model

# -----------------------------
# UI
# -----------------------------
st.title("Explainable Time-Series Anomaly Detection")

uploaded_file = st.file_uploader("Upload CSV file")

col1, col2 = st.columns(2)
with col1:
    seq_len = st.slider("Sequence Length", 10, 100, 30)
with col2:
    threshold_percentile = st.slider("Threshold Percentile", 80, 99, 95)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    values = df['value'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    # -----------------------------
    # Time Series Plot
    # -----------------------------
    st.subheader("Time Series Data")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(values_scaled)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Scaled Value")
    ax.set_title("Normalized Time Series")
    st.pyplot(fig)

    # -----------------------------
    # Prepare data
    # -----------------------------
    sequences = create_sequences(values_scaled, seq_len)
    sequences = sequences.squeeze()

    X = torch.tensor(sequences, dtype=torch.float32)

    st.write("Training model...")
    model = train_model(X, seq_len)
    st.success("Model training complete")

    # -----------------------------
    # Reconstruction
    # -----------------------------
    recon = model(X).detach().numpy()
    error = np.abs(sequences - recon)

    score = error.mean(axis=1)
    threshold = np.percentile(score, threshold_percentile)
    anomalies = score > threshold

    # -----------------------------
    # Metrics
    # -----------------------------
    st.subheader("Summary Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Points", len(values))
    col2.metric("Anomalies Detected", int(anomalies.sum()))
    col3.metric("Threshold", round(threshold, 4))

    # -----------------------------
    # Anomaly Score Plot
    # -----------------------------
    st.subheader("Anomaly Score")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(score, label="Reconstruction Error")
    ax.axhline(threshold, linestyle='--', label="Threshold")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Error")
    ax.set_title("Anomaly Score Over Time")
    ax.legend()
    st.pyplot(fig)

    # -----------------------------
    # Detected anomalies
    # -----------------------------
    anomaly_indices = np.where(anomalies)[0] + seq_len

    st.subheader("Detected Anomalies")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(values_scaled, label="Signal")
    ax.scatter(anomaly_indices, values_scaled[anomaly_indices], color='red', label="Anomaly")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Scaled Value")
    ax.set_title("Anomalies on Time Series")
    ax.legend()
    st.pyplot(fig)

    # -----------------------------
    # Heatmap
    # -----------------------------
    st.subheader("Reconstruction Error Heatmap")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.heatmap(error.T, cmap="Reds", ax=ax)
    ax.set_xlabel("Time Window Index")
    ax.set_ylabel("Sequence Step")
    ax.set_title("Error Distribution Across Time Windows")
    st.pyplot(fig)

    # -----------------------------
    # Anomaly-only heatmap
    # -----------------------------
    st.subheader("Anomaly-Focused Heatmap")

    anomaly_cols = np.where(anomalies)[0]

    if len(anomaly_cols) > 0:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        sns.heatmap(error.T[:, anomaly_cols], cmap="Reds", ax=ax)
        ax.set_xlabel("Anomalous Windows")
        ax.set_ylabel("Sequence Step")
        ax.set_title("Error in Anomalous Regions")
        st.pyplot(fig)

    # -----------------------------
    # Explainability
    # -----------------------------
    st.subheader("Explainability Analysis")

    if anomalies.any():
        top_idx = np.argmax(score)

        st.write(f"Highest anomaly detected at window index: {top_idx}")

        contrib = error[top_idx]
        important_step = np.argmax(contrib)

        st.write(f"Most significant deviation occurs at sequence step: {important_step}")

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(contrib)
        ax.set_xlabel("Sequence Step")
        ax.set_ylabel("Error Magnitude")
        ax.set_title("Error Contribution within Anomaly Window")
        st.pyplot(fig)

    else:
        st.write("No anomalies detected")