import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the synthetic test data
df = pd.read_csv("synthetic_balanced_test_data.csv")

# 2. Separate features and labels
X_test = df.drop(columns=["Grade"])
y_test = df["Grade"]

# 3. Standardize the test features (using fresh scaler)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)  # Ideally use the scaler from training

# 4. Load the trained model
model = load_model("glioma_grading_model_dense_smote.h5")

# 5. Predict
y_pred_probs = model.predict(X_test_scaled)
y_pred = (y_pred_probs > 0.5).astype(int)

# 6. Evaluate
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
