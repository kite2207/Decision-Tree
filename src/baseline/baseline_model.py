import pandas as pd
import joblib
from pathlib import Path


from sklearn.tree import DecisionTreeClassifier


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_FILE = BASE_DIR / "dataset/online_shoppers_train.csv"
TEST_FILE = BASE_DIR / "dataset/online_shoppers_test.csv"

MODEL_FILE = BASE_DIR / "model/baseline_model.joblib"
OUTPUT_FILE = BASE_DIR / "dataset/online_shoppers_baseline_predictions.csv"

TARGET = "Revenue"


# ============================================================
# 1. LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("=" * 70)
print("1. LOAD DATA")
print("=" * 70)

print(f"Train shape: {train_df.shape}")
print(f"Test shape : {test_df.shape}")


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])


print("\n" + "=" * 70)
print("2. DATA PREPARATION")
print("=" * 70)

print(f"Number of features : {X_train.shape[1]}")
print(f"Training samples   : {X_train.shape[0]}")
print(f"Testing samples    : {X_test.shape[0]}")


# ============================================================
# 3. CREATE BASELINE MODEL
# ============================================================

print("\n" + "=" * 70)
print("3. CREATE BASELINE MODEL")
print("=" * 70)

model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

print("Model              : DecisionTreeClassifier")
print("Criterion           : Gini")
print("Max depth           : None")
print("Min samples split   : 2")
print("Min samples leaf    : 1")


# ============================================================
# 4. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("4. TRAIN MODEL")
print("=" * 70)

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 5. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print(f"Model saved to: {MODEL_FILE}")


# ============================================================
# 6. PREDICT TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("5. PREDICT TEST DATA")
print("=" * 70)

y_pred = model.predict(X_test)

print(f"Predictions generated: {len(y_pred)}")


# ============================================================
# 7. SAVE PREDICTIONS
# ============================================================

result = test_df.copy()

result["Predicted_Revenue"] = y_pred

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Predictions saved to: {OUTPUT_FILE}")


# ============================================================
# 8. MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("6. MODEL INFORMATION")
print("=" * 70)

print(f"Tree depth       : {model.get_depth()}")
print(f"Number of leaves : {model.get_n_leaves()}")
print(f"Number of nodes  : {model.tree_.node_count}")


print("\nBaseline completed successfully.")