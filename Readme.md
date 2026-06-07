# 🛡️ Credit Card Fraud Detection Intelligence Dashboard

An interactive Streamlit web application that utilizes a Machine Learning Decision Tree Classifier to analyze, detect, and rank fraudulent credit card transactions in real-time based on the Kaggle Credit Card Fraud dataset.

## 🚀 Features & Interface

### 1. High-Level KPIs & Performance Diagnostics
- Displays crucial production metrics instantly: **Accuracy (99.93%)**, **Precision (0.830)**, and **Recall (0.745)**.
- **ROC Curve:** Features an interactive Receiver Operating Characteristic curve hitting a robust **AUC of 0.932**.
- **Feature Importance:** Automatically ranks mathematical features, proving that feature `V17` is the single largest indicator of fraudulent patterns.

![Dashboard Performance Metrics and Diagnostics](dashboard_metrics.jpg)

---

### 2. Live Operational Risk Queue
- **Probability Scoring:** Moves past binary "yes/no" classifications by extracting raw leaf-node probabilities.
- **Dynamic Threat Tiers:** Automatically flags events as *Critical*, *Medium*, or *Low* risk.
- **Analyst-First Sorting:** Floats the highest risk transactions straight to the top of the queue for investigation.

![Real-Time Interactive Fraud Risk Queue Table](risk_queue.jpg)

---

## 📦 Installation & Setup
1. Clone this repository to your local machine.
2. Download the `creditcard.csv` dataset from Kaggle and place it in the project root folder.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt