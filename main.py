import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = "online_shoppers_train.csv"
TEST_FILE = "online_shoppers_test.csv"

MODEL_FILE = "baseline_model.joblib"


TARGET = "Revenue"


# ============================================================
# 1. LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


# ============================================================
# 2. LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_FILE
)


# ============================================================
# 3. PREPARE DATA
# ============================================================

X_train = train_df.drop(
    columns=[TARGET]
)

y_train = train_df[TARGET]

X_test = test_df.drop(
    columns=[TARGET]
)

y_test = test_df[TARGET]


# ============================================================
# 4. PREDICT
# ============================================================

y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)


# ============================================================
# 5. MODEL INFORMATION
# ============================================================

print("=" * 70)
print("BASELINE DECISION TREE")
print("=" * 70)

print("\nMODEL PARAMETERS")

print(f"Criterion         : {model.criterion}")
print(f"Max depth         : {model.max_depth}")
print(f"Min samples split : {model.min_samples_split}")
print(f"Min samples leaf  : {model.min_samples_leaf}")
print(f"Random state      : {model.random_state}")


print("\nTREE INFORMATION")

print(f"Tree depth        : {model.get_depth()}")
print(f"Number of leaves  : {model.get_n_leaves()}")
print(f"Number of nodes   : {model.tree_.node_count}")


# ============================================================
# 6. DATA INFORMATION
# ============================================================

print("\nDATA")

print(f"Training samples  : {len(X_train)}")
print(f"Testing samples   : {len(X_test)}")
print(f"Features          : {X_train.shape[1]}")


# ============================================================
# 7. TRAINING PERFORMANCE
# ============================================================

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)


# ============================================================
# 8. TEST PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

error_rate = 1 - accuracy


# ============================================================
# 9. PRINT METRICS
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE")
print("=" * 70)

print(f"Training Accuracy : {train_accuracy:.4f}")

print(f"\nTesting Accuracy  : {accuracy:.4f}")
print(f"Error Rate        : {error_rate:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"F1-score          : {f1:.4f}")


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("                 Predicted")
print("                 No    Yes")

print(
    f"Actual No      {cm[0][0]:5d} {cm[0][1]:5d}"
)

print(
    f"Actual Yes     {cm[1][0]:5d} {cm[1][1]:5d}"
)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=[
            "No Purchase",
            "Purchase"
        ],
        zero_division=0
    )
)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 70)
print("TOP 15 FEATURE IMPORTANCE")
print("=" * 70)

print(
    feature_importance
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 13. FINAL
# ============================================================

print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)