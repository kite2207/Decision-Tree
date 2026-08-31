import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "dataset/online_shoppers_intention.csv"

TRAIN_FILE = BASE_DIR / "dataset/online_shoppers_train.csv"
TEST_FILE = BASE_DIR / "dataset/online_shoppers_test.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("1. LOAD DATA")
print("=" * 70)

print(f"Original rows    : {df.shape[0]}")
print(f"Original columns : {df.shape[1]}")


# ============================================================
# 2. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("2. MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)

print(f"\nTotal missing values: {missing.sum()}")

if missing.sum() == 0:
    print("No missing values found.")
else:
    print("Missing values found.")


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("3. REMOVE DUPLICATES")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count}")

df = df.drop_duplicates().reset_index(drop=True)

print(f"Rows after removing duplicates: {len(df)}")


# ============================================================
# 4. DEFINE FEATURES
# ============================================================

numerical_features = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

categorical_features = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType"
]

binary_features = [
    "Weekend"
]

TARGET = "Revenue"


# ============================================================
# 5. SEPARATE X AND y
# ============================================================

X = df.drop(columns=[TARGET])

# False -> 0
# True  -> 1
y = df[TARGET].astype(int)


# ============================================================
# 6. CHECK TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("4. TARGET DISTRIBUTION")
print("=" * 70)

target_counts = y.value_counts().sort_index()
target_percent = y.value_counts(
    normalize=True
).sort_index() * 100

for value in target_counts.index:

    if value == 0:
        label = "No Purchase"
    else:
        label = "Purchase"

    print(
        f"{label:<15}: "
        f"{target_counts[value]:>5} "
        f"({target_percent[value]:.2f}%)"
    )


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("5. TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# 8. ONE-HOT ENCODING
# ============================================================

print("\n" + "=" * 70)
print("6. ONE-HOT ENCODING")
print("=" * 70)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# IMPORTANT:
# Fit encoder ONLY on training data.
# Test data uses the same encoder.

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ============================================================
# 9. FEATURE NAMES
# ============================================================

feature_names = preprocessor.get_feature_names_out()

print(f"Original features : {X.shape[1]}")
print(f"Processed features: {len(feature_names)}")


# ============================================================
# 10. CONVERT TO DATAFRAME
# ============================================================

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)


# ============================================================
# 11. ADD TARGET
# ============================================================

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

X_train_processed["Revenue"] = y_train
X_test_processed["Revenue"] = y_test


# ============================================================
# 12. SAVE TRAIN / TEST CSV
# ============================================================

X_train_processed.to_csv(
    TRAIN_FILE,
    index=False
)

X_test_processed.to_csv(
    TEST_FILE,
    index=False
)


# ============================================================
# 13. VALIDATE OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("7. OUTPUT FILES")
print("=" * 70)

print(f"Train file: {TRAIN_FILE}")
print(f"Test file : {TEST_FILE}")

print(f"\nTrain shape: {X_train_processed.shape}")
print(f"Test shape : {X_test_processed.shape}")


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)

print(f"Original dataset : 12330 rows")
print(f"Duplicates       : {duplicate_count}")
print(f"Final dataset    : {len(df)} rows")

print(f"\nOriginal features: {X.shape[1]}")
print(f"Encoded features : {len(feature_names)}")

print(f"\nTrain samples    : {len(X_train_processed)}")
print(f"Test samples     : {len(X_test_processed)}")

print("\nGenerated files:")
print(f"  - {TRAIN_FILE}")
print(f"  - {TEST_FILE}")

print("\nPreprocessing completed successfully.")