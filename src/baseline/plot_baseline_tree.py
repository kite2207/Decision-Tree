import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.tree import plot_tree


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_FILE = BASE_DIR / "model/baseline_model.joblib"
TRAIN_FILE = BASE_DIR / "dataset/online_shoppers_train.csv"

OUTPUT_IMAGE = BASE_DIR / "img/baseline_decision_tree_top3.png"

TARGET = "Revenue"

# Chỉ hiển thị 3 tầng đầu của cây
DISPLAY_DEPTH = 3


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("=" * 70)
print("1. LOAD BASELINE MODEL")
print("=" * 70)

model = joblib.load(MODEL_FILE)

print(f"Model loaded: {MODEL_FILE}")


# ============================================================
# 2. LOAD FEATURE NAMES
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)

X_train = train_df.drop(
    columns=[TARGET]
)

feature_names = X_train.columns


# ============================================================
# 3. MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("2. MODEL INFORMATION")
print("=" * 70)

print(f"Full tree depth       : {model.get_depth()}")
print(f"Full tree leaves      : {model.get_n_leaves()}")
print(f"Full tree nodes       : {model.tree_.node_count}")

print(f"\nDisplayed depth       : {DISPLAY_DEPTH}")


# ============================================================
# 4. DRAW TREE
# ============================================================

print("\n" + "=" * 70)
print("3. DRAW DECISION TREE")
print("=" * 70)

plt.figure(
    figsize=(24, 14)
)

plot_tree(
    model,
    max_depth=DISPLAY_DEPTH,
    feature_names=feature_names,
    class_names=[
        "No Purchase",
        "Purchase"
    ],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title(
    "Baseline Decision Tree - First 3 Levels",
    fontsize=18
)

plt.tight_layout()

plt.savefig(
    OUTPUT_IMAGE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Image saved: {OUTPUT_IMAGE}")