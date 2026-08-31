"""
Improvement 3: Compare Gini vs Entropy splitting criterion
Person 3 - Day 3
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR   = Path(__file__).resolve().parents[2]
TRAIN_FILE = BASE_DIR / "dataset" / "online_shoppers_train.csv"
TEST_FILE  = BASE_DIR / "dataset" / "online_shoppers_test.csv"
OUTPUT_TXT = BASE_DIR / "doc" / "improvement3_result.txt"
OUTPUT_IMG = BASE_DIR / "img" / "improvement3_best_tree.png"

TARGET     = "Revenue"

# ============================================================
# 1. LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df  = pd.read_csv(TEST_FILE)

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET]

feature_names = list(X_train.columns)
class_names   = ["No Purchase", "Purchase"]

# ============================================================
# 2. TRAIN AND EVALUATE
# ============================================================

def evaluate(criterion):
    clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test  = clf.predict(X_test)

    return {
        "model"         : clf,
        "criterion"     : criterion,
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "accuracy"      : accuracy_score(y_test, y_pred_test),
        "error_rate"    : 1 - accuracy_score(y_test, y_pred_test),
        "precision"     : precision_score(y_test, y_pred_test, zero_division=0),
        "recall"        : recall_score(y_test, y_pred_test, zero_division=0),
        "f1"            : f1_score(y_test, y_pred_test, zero_division=0),
        "depth"         : clf.get_depth(),
        "leaves"        : clf.get_n_leaves(),
        "nodes"         : clf.tree_.node_count,
        "cm"            : confusion_matrix(y_test, y_pred_test),
        "report"        : classification_report(
                              y_test, y_pred_test,
                              target_names=["No Purchase", "Purchase"],
                              zero_division=0
                          ),
        "y_pred"        : y_pred_test,
    }

gini_res    = evaluate("gini")
entropy_res = evaluate("entropy")
results     = [gini_res, entropy_res]

# ============================================================
# 3. BUILD REPORT
# ============================================================

lines = []
lines.append("=" * 70)
lines.append("IMPROVEMENT 3: GINI vs ENTROPY - COMPARISON RESULTS")
lines.append("=" * 70)
lines.append("")
lines.append("Dataset : online_shoppers_intention (preprocessed)")
lines.append(f"Train   : {len(X_train)} samples")
lines.append(f"Test    : {len(X_test)} samples")
lines.append(f"Features: {X_train.shape[1]}")
lines.append("")

# Comparison table
lines.append("--- COMPARISON TABLE ---")
lines.append("")
header = f"{'Criterion':<12} {'Train Acc':>10} {'Test Acc':>10} {'Error':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Depth':>7} {'Leaves':>8} {'Nodes':>7}"
lines.append(header)
lines.append("-" * len(header))
for r in results:
    lines.append(
        f"{r['criterion']:<12} "
        f"{r['train_accuracy']:>10.4f} "
        f"{r['accuracy']:>10.4f} "
        f"{r['error_rate']:>8.4f} "
        f"{r['precision']:>10.4f} "
        f"{r['recall']:>8.4f} "
        f"{r['f1']:>8.4f} "
        f"{r['depth']:>7d} "
        f"{r['leaves']:>8d} "
        f"{r['nodes']:>7d}"
    )
lines.append("")

# Difference
diff_acc = entropy_res["accuracy"] - gini_res["accuracy"]
diff_f1  = entropy_res["f1"]       - gini_res["f1"]
lines.append(f"Accuracy difference (Entropy - Gini) : {diff_acc:+.4f}")
lines.append(f"F1-score difference (Entropy - Gini) : {diff_f1:+.4f}")
lines.append("")

# Best
best = max(results, key=lambda x: x["f1"])
lines.append(f"Best criterion (by F1-score) : {best['criterion'].upper()}")
lines.append("")

# Per-criterion detail
for r in results:
    lines.append("=" * 70)
    lines.append(f"DETAIL: {r['criterion'].upper()}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Training Accuracy : {r['train_accuracy']:.4f}")
    lines.append(f"Testing Accuracy  : {r['accuracy']:.4f}")
    lines.append(f"Error Rate        : {r['error_rate']:.4f}")
    lines.append(f"Precision         : {r['precision']:.4f}")
    lines.append(f"Recall            : {r['recall']:.4f}")
    lines.append(f"F1-score          : {r['f1']:.4f}")
    lines.append(f"Tree depth        : {r['depth']}")
    lines.append(f"Leaf nodes        : {r['leaves']}")
    lines.append(f"Total nodes       : {r['nodes']}")
    lines.append("")
    cm = r["cm"]
    lines.append("Confusion Matrix:")
    lines.append("                 Predicted No    Predicted Yes")
    lines.append(f"Actual No      {cm[0][0]:14d} {cm[0][1]:13d}")
    lines.append(f"Actual Yes     {cm[1][0]:14d} {cm[1][1]:13d}")
    lines.append("")
    lines.append("Classification Report:")
    lines.append(r["report"])

# Analysis
lines.append("=" * 70)
lines.append("ANALYSIS")
lines.append("=" * 70)
lines.append("")
lines.append("1. Gini Impurity:")
lines.append("   - Measures how often a randomly chosen element would be")
lines.append("     incorrectly classified.")
lines.append("   - Computationally faster (no log computation).")
lines.append("   - Tends to produce slightly larger/deeper trees.")
lines.append("")
lines.append("2. Entropy (Information Gain):")
lines.append("   - Measures information gained by a split.")
lines.append("   - Slightly slower due to log computation.")
lines.append("   - May produce more balanced splits on some datasets.")
lines.append("")
if abs(diff_acc) < 0.001:
    lines.append("3. Conclusion:")
    lines.append("   On this dataset, Gini and Entropy produce nearly identical")
    lines.append("   results. This is common for large, balanced preprocessing.")
    lines.append(f"   The difference in accuracy is only {abs(diff_acc)*100:.2f}%.")
    lines.append("   Both criteria result in trees of the same depth,")
    lines.append("   suggesting the dataset structure drives the splits more")
    lines.append("   than the criterion choice.")
    lines.append(f"   Recommended: {best['criterion'].upper()} (marginally better F1).")
elif best["criterion"] == "entropy":
    lines.append("3. Conclusion:")
    lines.append("   Entropy outperforms Gini on this dataset.")
    lines.append(f"   Accuracy gain: {abs(diff_acc)*100:.2f}%, F1 gain: {abs(diff_f1)*100:.2f}%.")
    lines.append("   The information gain criterion finds better splits")
    lines.append("   for this imbalanced classification problem.")
else:
    lines.append("3. Conclusion:")
    lines.append("   Gini outperforms Entropy on this dataset.")
    lines.append(f"   Accuracy gain: {abs(diff_acc)*100:.2f}%, F1 gain: {abs(diff_f1)*100:.2f}%.")
    lines.append("   Gini is both faster and produces better results here.")

output = "\n".join(lines)
print(output)
OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_TXT.write_text(output, encoding="utf-8")
print(f"\nSaved: {OUTPUT_TXT}")

# ============================================================
# 4. EXPORT BEST TREE IMAGE
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(32, 12))

for ax, r in zip(axes, results):
    plot_tree(
        r["model"], max_depth=4,
        feature_names=feature_names,
        class_names=class_names,
        filled=True, rounded=True,
        fontsize=7, ax=ax,
        impurity=True, proportion=False,
    )
    marker = " [BEST]" if r["criterion"] == best["criterion"] else ""
    ax.set_title(
        f"Criterion: {r['criterion'].upper()}{marker}\n"
        f"Accuracy={r['accuracy']:.4f}  F1={r['f1']:.4f}  "
        f"Depth={r['depth']}  Leaves={r['leaves']}",
        fontsize=11, fontweight="bold"
    )

fig.suptitle(
    "Improvement 3: Gini vs Entropy (Top 4 Levels)",
    fontsize=14, fontweight="bold", y=1.01
)
fig.tight_layout()
OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_IMG, dpi=100, bbox_inches="tight")
plt.close(fig)
print(f"Image saved: {OUTPUT_IMG}")
print("\nDone.")
