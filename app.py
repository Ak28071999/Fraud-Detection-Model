import numpy as np 
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve
)

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Fraud Intelligence Dashboard", layout="wide")
st.title("🛡️ Fraud Intelligence: Decision Tree Structure")
st.markdown("This dashboard visualizes the internal logic and performance of a Fraud Detection model.")

# --- 2. CACHED DATA PIPELINE ---
@st.cache_resource
def run_full_pipeline():
    # Load dataset
    df = pd.read_csv("creditcard.csv")
    
    # Feature Engineering: Drop target and low-importance features
    y = df["Class"]
    X = df.drop(["Class", "V23", "V18", "V9", "V2", "V5", "V10", "V27", "V24", "V8"], axis=1)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    dt_model = DecisionTreeClassifier(max_depth=7, min_samples_split=50, random_state=42)
    dt_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = dt_model.predict(X_test_scaled)
    y_probs = dt_model.predict_proba(X_test_scaled)[:, 1]
    
    results = {
        "acc": accuracy_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred),
        "rec": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_probs),
        "cm": confusion_matrix(y_test, y_pred),
        "cr": classification_report(y_test, y_pred)
    }
    
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    
    # Returning the model inputs and original dataframe indices to reconstruct the queue down below
    return dt_model, X.columns, results, fpr, tpr, y_test, X_test_scaled, df

# Initialize Pipeline
model, feature_names, metrics, fpr, tpr, y_test, X_test_scaled, df_original = run_full_pipeline()

# --- 3. KPI ROW ---
cols = st.columns(5)
cols[0].metric("Accuracy", f"{metrics['acc']*100:.2f}%")
cols[1].metric("Precision", f"{metrics['prec']:.3f}")
cols[2].metric("Recall", f"{metrics['rec']:.3f}")
cols[3].metric("F1 Score", f"{metrics['f1']:.3f}")
cols[4].metric("ROC AUC", f"{metrics['auc']:.3f}")

st.divider()

# --- 4. DIAGNOSTIC PLOTS ---
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📈 ROC Curve")
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    ax_roc.plot(fpr, tpr, color="#deff9a", lw=2, label=f"AUC: {metrics['auc']:.3f}")
    ax_roc.plot([0, 1], [0, 1], color="#333", linestyle="--")
    ax_roc.set_title("Model Sensitivity vs Specificity")
    ax_roc.legend()
    st.pyplot(fig_roc)

with col_b:
    st.subheader("📊 Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": feature_names, 
        "Weight": model.feature_importances_
    }).sort_values("Weight", ascending=False).head(10)
    
    fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
    sns.barplot(data=importance_df, x="Weight", y="Feature", palette="viridis", ax=ax_imp)
    st.pyplot(fig_imp)

st.divider()

# --- 5. THE STRUCTURE CHART (Visual Logic) ---
st.subheader("🌲 Decision Logic: Tree Structure Chart")
st.info("Showing the top 3 levels of the model's decision architecture.")

plt.clf()
plt.close('all')

fig_tree = plt.figure(figsize=(24, 12))
ax_tree = fig_tree.add_subplot(111)

clean_feature_names = [str(col) for col in feature_names]

plot_tree(
    model,
    feature_names=clean_feature_names,
    class_names=["Normal", "Fraud"],
    filled=True,
    rounded=True,
    max_depth=7,  
    fontsize=11,
    ax=ax_tree
)

st.pyplot(fig_tree)

st.divider()

# --- 6. FRAUD RISK QUEUE MANAGEMENT ---
st.subheader("🔍 Real-Time Fraud Risk Queue")
st.markdown("This table ranks all testing transactions by their calculated probability of fraud.")

# Extract probabilities for the testing matrix
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# Rebuild an analyst display DataFrame using indices matched with our original file rows
risk_queue = pd.DataFrame(index=y_test.index)
risk_queue["Transaction_Index"] = y_test.index
risk_queue["Actual_Class"] = y_test.map({0: "Normal", 1: "🚨 FRAUD"})
risk_queue["Fraud_Probability"] = y_probs

# Assign dynamic warning flags based on calculated threat metrics
def assign_risk_flag(prob):
    if prob >= 0.75:
        return "🔴 Critical Risk"
    elif prob >= 0.25:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"

risk_queue["Risk_Tier"] = risk_queue["Fraud_Probability"].apply(assign_risk_flag)

# Inject transaction 'Amount' value back into the queue for investigator visibility
if "Amount" in df_original.columns:
    risk_queue["Amount"] = df_original.loc[y_test.index, "Amount"]

# Sort the table so the absolute highest risks float right to the top
risk_queue = risk_queue.sort_values(by="Fraud_Probability", ascending=False)

# Render interactive dataframe with a progress bar container component
st.dataframe(
    risk_queue.reset_index(drop=True),
    use_container_width=True,
    column_config={
        "Fraud_Probability": st.column_config.ProgressColumn(
            "Fraud Probability Score",
            help="The probability score assigned by the final leaf nodes",
            format="%.2f",
            min_value=0.0,
            max_value=1.0,
        ),
        "Amount": st.column_config.NumberColumn(
            "Transaction Amount ($)",
            format="$%.2f"
        )
    }
)

st.divider()

# --- 7. RAW REPORTS ---
with st.expander("View Full Classification Report"):
    st.code(metrics["cr"])