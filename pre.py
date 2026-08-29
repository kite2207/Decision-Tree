import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "online_shoppers_intention.csv"
OUTPUT_FILE = "online_shoppers_preprocessed.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("1. ORIGINAL DATASET")
print("=" * 70)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 2. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("2. MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)
print(f"\nTotal missing values: {missing.sum()}")


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("3. DUPLICATES")
print("=" * 70)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")

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

target = "Revenue"


# ============================================================
# 5. SEPARATE X AND y
# ============================================================

X = df.drop(columns=[target])
y = df[target].astype(int)


# ============================================================
# 6. DISPLAY TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("4. TARGET DISTRIBUTION")
print("=" * 70)

counts = y.value_counts()
percentages = y.value_counts(normalize=True) * 100

for value in sorted(counts.index):

    if value == 0:
        label = "No Purchase"
    else:
        label = "Purchase"

    print(
        f"{label:<15}: "
        f"{counts[value]:>5} "
        f"({percentages[value]:.2f}%)"
    )


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("5. TRAIN / TEST SPLIT")
print("=" * 70)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# 8. ONE-HOT ENCODING
# ============================================================

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
# Fit encoder ONLY using training data.
# Then use the same encoder for test data.

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ============================================================
# 9. GET FEATURE NAMES
# ============================================================

feature_names = preprocessor.get_feature_names_out()


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
# 12. ADD TRAIN / TEST LABEL
# ============================================================

X_train_processed["Split"] = "train"
X_test_processed["Split"] = "test"


# ============================================================
# 13. COMBINE TRAIN + TEST
# ============================================================

processed_data = pd.concat(
    [
        X_train_processed,
        X_test_processed
    ],
    ignore_index=True
)


# Put Split and Revenue at the end

feature_columns = [
    column
    for column in processed_data.columns
    if column not in ["Split", "Revenue"]
]

processed_data = processed_data[
    feature_columns + ["Split", "Revenue"]
]


# ============================================================
# 14. SAVE CSV
# ============================================================

processed_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("6. PREPROCESSING RESULT")
print("=" * 70)

print(f"Original records       : {df.shape[0] + duplicates}")
print(f"Duplicates removed     : {duplicates}")
print(f"Final records          : {df.shape[0]}")

print(f"\nOriginal features      : {X.shape[1]}")
print(f"Processed features     : {len(feature_names)}")

print(f"\nTraining samples       : {len(X_train_processed)}")
print(f"Testing samples        : {len(X_test_processed)}")

print(f"\nOutput file            : {OUTPUT_FILE}")
print(f"Output shape           : {processed_data.shape}")


# ============================================================
# 16. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("7. VALIDATION")
print("=" * 70)

print(
    "Missing values:",
    processed_data.isnull().sum().sum()
)

print(
    "Train rows:",
    (processed_data["Split"] == "train").sum()
)

print(
    "Test rows:",
    (processed_data["Split"] == "test").sum()
)

print("\nPreprocessing completed successfully.")