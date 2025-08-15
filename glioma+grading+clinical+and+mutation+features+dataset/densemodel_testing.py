import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load the same top 8 features as used in training ---
top_8_features = ['IDH1', 'Age_at_diagnosis', 'PIK3CA', 'ATRX', 'PTEN', 'CIC', 'EGFR', 'TP53']

# --- 2. Load new test data (100 samples synthetic or real) ---
new_df = pd.read_csv("synthetic_balanced_test_data.csv")
X_new = new_df[top_8_features]
y_true = new_df["Grade"]  # if available

# --- 3. Load model and prepare data ---
model = load_model("glioma_grading_model_optimized.h5")

# Refit StandardScaler (you must ensure scaling is consistent)
scaler = StandardScaler()
scaler.fit(pd.read_csv("TCGA_InfoWithGrade.csv")[top_8_features])  # Use original training stats
X_scaled = scaler.transform(X_new)

# --- 4. Predict ---
y_probs = model.predict(X_scaled).ravel()

# ROC-based threshold (same as used during training)
threshold = 0.45
y_pred = (y_probs > threshold).astype(int)

# --- 5. Results ---
results_df = new_df.copy()
results_df["Prediction"] = y_pred
results_df["Confidence"] = y_probs
print("\n🧠 Prediction Results:")
print(results_df[["Prediction", "Confidence"]].head())

# --- 6. Evaluation if true labels are available ---
if "Grade" in new_df.columns:
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# --- 7. Save predictions ---
results_df.to_csv("predicted_results.csv", index=False)
print("\n✅ Predictions saved to 'predicted_results.csv'")
