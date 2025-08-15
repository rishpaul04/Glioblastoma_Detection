import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
df = pd.read_csv("TCGA_InfoWithGrade.csv")
top_8_features = ['IDH1', 'Age_at_diagnosis', 'PIK3CA', 'ATRX', 'PTEN', 'CIC', 'EGFR', 'TP53']
X = df[top_8_features]
y = df["Grade"]  # 0 = LGG, 1 = GBM

# 2. Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. SMOTE to balance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 5. Build model with regularization
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(alpha=0.1),
    Dropout(0.4),

    Dense(128, kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(alpha=0.1),
    Dropout(0.4),

    Dense(64, kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=16)

# 7. Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.2f}")

# 8. Predict and threshold tuning
y_probs = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], '--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.show()

# Choose best threshold (You can tune this manually based on curve or metrics)
threshold = 0.45
y_pred = (y_probs > threshold).astype(int)

# 9. Classification metrics
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 10. Save model and scaler
model.save("glioma_grading_model_optimized.h5")
pd.DataFrame(scaler.mean_, index=top_8_features).to_csv("scaler_mean.csv")
print("✅ Model saved as 'glioma_grading_model_optimized.h5'")
