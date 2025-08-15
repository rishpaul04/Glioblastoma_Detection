import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# 1. Load and preprocess data
df = pd.read_csv("TCGA_InfoWithGrade.csv")
X = df.drop(columns=["Grade"])
y = df["Grade"]  # binary: 0 = LGG, 1 = GBM

# Keep only numeric columns (ignore string IDs etc.)
X = X.select_dtypes(include=[np.number])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for Conv1D: (samples, features, 1)
X_reshaped = np.expand_dims(X_scaled, axis=2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# 2. Handle class imbalance
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# 3. Define CNN model
model = Sequential([
    Conv1D(32, kernel_size=3, padding='valid', input_shape=(X_train.shape[1], 1)),
    LeakyReLU(alpha=0.1),

    Conv1D(64, kernel_size=3, padding='valid'),
    LeakyReLU(alpha=0.1),

    Conv1D(128, kernel_size=3, padding='valid'),
    LeakyReLU(alpha=0.1),

    MaxPooling1D(pool_size=3),
    Dropout(0.5),
    Flatten(),

    Dense(256),
    LeakyReLU(alpha=0.1),

    Dense(512),
    LeakyReLU(alpha=0.1),

    Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train model with EarlyStopping and Class Weights
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights_dict,
    callbacks=[early_stop]
)

# 5. Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.2f}")

# 6. Plot training results
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# 7. Confusion matrix and report
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 8. Save model
model.save("glioma_grading_model_balanced.h5")
print("✅ Model saved as 'glioma_grading_model_balanced.h5'")
