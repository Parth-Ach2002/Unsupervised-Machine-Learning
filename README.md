# Unsupervised-Machine-Learning

# 🚨 Explainable Time-Series Anomaly Detection using Autoencoders

## 📌 Project Description

This project presents an **unsupervised anomaly detection system for time-series data** using an autoencoder-based neural network. The model learns normal patterns in sequential data and identifies anomalies based on reconstruction error. To enhance usability and interpretability, the system is deployed as an **interactive Streamlit web application**, allowing real-time visualization, parameter tuning, and dataset upload.

The approach is evaluated on real-world datasets and provides both **accurate detection** and **explainability**, making it suitable for applications such as traffic monitoring, occupancy analysis, and IoT systems.

---

## 🎯 Objectives

* Detect anomalies in time-series data without labeled training data
* Learn temporal patterns using an autoencoder model
* Provide explainable insights into detected anomalies
* Build an interactive dashboard for real-time visualization

---

## 📊 Datasets Used

### 🚗 Speed Dataset

* Real-world traffic speed data
* Captures variations in vehicle speed over time
* Contains anomalies such as sudden drops/spikes

### 🏢 Occupancy Dataset

* Sensor-based occupancy-related data
* Reflects changes in environmental patterns
* Used to detect irregular occupancy behavior

📥 Source:
[https://github.com/numenta/NAB](https://github.com/numenta/NAB)

---

## ⚙️ Methodology

1. **Data Preprocessing**

   * Convert timestamps
   * Normalize values using MinMaxScaler

2. **Sequence Generation**

   * Sliding window approach
   * Converts time-series into sequences

3. **Model Training**

   * Autoencoder neural network
   * Learns normal patterns using reconstruction loss

4. **Anomaly Detection**

   * Compute reconstruction error
   * Apply percentile-based threshold

5. **Visualization**

   * Time series plots
   * Anomaly score graphs
   * Heatmaps
   * Explainability plots

---

## 🧠 Explainability

The model includes an explainability module that:

* Identifies the most anomalous window
* Highlights which timestep contributes most to the anomaly
* Uses heatmaps to visualize reconstruction error

---

## 📊 Results

The model was tested on both Speed and Occupancy datasets and produced consistent results.

* Successfully detected anomalies as **peaks in reconstruction error**
* Identified abnormal regions in time-series plots
* Heatmaps clearly highlighted high-error regions
* Explainability plots showed **which part of the sequence caused anomalies**

### Key Observations:

* Higher threshold → fewer but stronger anomalies
* Lower threshold → more sensitive detection
* Larger sequence length → captures long-term patterns
* Smaller sequence length → detects short-term variations

---

## 🖥️ Streamlit Dashboard Features

* Upload custom CSV datasets
* Adjustable parameters:

  * Sequence Length
  * Threshold Percentile
* Real-time model retraining
* Interactive visualizations:

  * Time Series Plot
  * Anomaly Score
  * Detected Anomalies
  * Heatmaps
  * Explainability Graph

---

## 📦 Installation & Setup

```bash
# Clone repository
git clone https://github.com/your-username/anomaly-detection.git

# Navigate to project
cd anomaly-detection

# Install dependencies
pip install pandas numpy torch scikit-learn matplotlib seaborn streamlit

# Run the app
python -m streamlit run app.py
```

---

## 📁 Project Structure

```
📦 anomaly-detection
 ┣ 📜 app.py
 ┣ 📜 README.md
 ┣ 📂 data
 ┃ ┣ speed_6005.csv
 ┃ ┗ occupancy.csv
 ┗ 📜 requirements.txt

## 🔮 Future Scope

* Integrate LSTM or VAE for better temporal modeling
* Use adaptive thresholding instead of fixed percentile
* Extend to multivariate time-series data
* Deploy as a cloud-based real-time monitoring system

---

## 📚 References

* [https://github.com/numenta/NAB](https://github.com/numenta/NAB)
* Goodfellow et al., *Deep Learning*, MIT Press
* [https://docs.streamlit.io](https://docs.streamlit.io)
* [https://pytorch.org](https://pytorch.org)

## 👨‍💻 Author

**Parth Achrekar**

## ⭐ Acknowledgment

This project demonstrates the application of **unsupervised learning and explainable AI** for real-world anomaly detection problems.

If you want:
✅ I can **add images section properly formatted**
✅ or generate a **perfect GitHub repo description + tags**

Just tell me 👍
