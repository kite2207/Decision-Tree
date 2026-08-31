import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import plot_tree

BASE_DIR    = Path(__file__).resolve().parents[2]
MODEL_FILE  = BASE_DIR / "model" / "baseline_model.joblib"
TRAIN_FILE  = BASE_DIR / "dataset" / "online_shoppers_train.csv"
OUTPUT_IMG  = BASE_DIR / "img" / "baseline_decision_tree_top5.png"
OUTPUT_TXT  = BASE_DIR / "doc" / "tree_analysis_raw.txt"

model    = joblib.load(MODEL_FILE)
train_df = pd.read_csv(TRAIN_FILE)

TARGET        = "Revenue"
feature_names = [c for c in train_df.columns if c != TARGET]
class_names   = ["No Purchase", "Purchase"]

depth  = model.get_depth()
leaves = model.get_n_leaves()
nodes  = model.tree_.node_count
tree_  = model.tree_

root_feature = feature_names[tree_.feature[0]]
root_thresh  = tree_.threshold[0]

fi_df = (
    pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)
used_features = fi_df[fi_df["Importance"] > 0]

lines = []
lines.append("=" * 70)
lines.append("BASELINE DECISION TREE - STRUCTURAL ANALYSIS")
lines.append("=" * 70)
lines.append("")
lines.append("--- TREE SIZE ---")
lines.append(f"Max depth        : {depth}")
lines.append(f"Total nodes      : {nodes}")
lines.append(f"Leaf nodes       : {leaves}")
lines.append(f"Internal nodes   : {nodes - leaves}")
lines.append("")
lines.append("--- ROOT NODE ---")
lines.append(f"Split feature    : {root_feature}")
lines.append(f"Split threshold  : {root_thresh:.6f}")
lines.append(f"  Left  branch   : {root_feature} <= {root_thresh:.4f}")
lines.append(f"  Right branch   : {root_feature}  > {root_thresh:.4f}")
lines.append("")
lines.append("--- TOP 10 MOST IMPORTANT FEATURES ---")
for i, row in fi_df.head(10).iterrows():
    lines.append(f"  {i+1:2d}. {row['Feature']:<35s} {row['Importance']:.6f}")
lines.append("")
lines.append(f"Total features used in splits: {len(used_features)}")
lines.append(f"Total features available     : {len(feature_names)}")

def collect_path(tree_, feature_names, class_names, go_left_at_root=True):
    steps = []
    node  = 0
    d     = 0
    while tree_.feature[node] != -2:
        feat   = feature_names[tree_.feature[node]]
        thresh = tree_.threshold[node]
        go_left = go_left_at_root if d == 0 else True
        direction = "<=" if go_left else " >"
        steps.append(f"{'  ' * d}Node {node}: {feat} {direction} {thresh:.4f}")
        node = tree_.children_left[node] if go_left else tree_.children_right[node]
        d += 1
    values     = tree_.value[node][0]
    pred_class = class_names[int(values.argmax())]
    total      = int(values.sum())
    steps.append(
        f"{'  ' * d}-> LEAF Node {node}: predict '{pred_class}' "
        f"(No Purchase={int(values[0])}, Purchase={int(values[1])}, n={total})"
    )
    return steps

lines.append("")
lines.append("--- PATH TRACING ---")
lines.append("")
lines.append("Path 1: Go LEFT at root (condition TRUE -> smaller/equal threshold)")
lines.extend(collect_path(tree_, feature_names, class_names, go_left_at_root=True))
lines.append("")
lines.append("Path 2: Go RIGHT at root (condition FALSE -> greater than threshold)")
lines.extend(collect_path(tree_, feature_names, class_names, go_left_at_root=False))

output = "\n".join(lines)
print(output)
OUTPUT_TXT.write_text(output, encoding="utf-8")
print(f"\nSaved to: {OUTPUT_TXT}")

fig, ax = plt.subplots(figsize=(28, 10))
plot_tree(
    model, max_depth=5,
    feature_names=feature_names,
    class_names=class_names,
    filled=True, rounded=True,
    fontsize=8, ax=ax,
    impurity=True, proportion=False,
)
ax.set_title("Baseline Decision Tree - Top 5 Levels", fontsize=14, fontweight="bold", pad=12)
fig.tight_layout()
OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_IMG, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Top-5 image saved to: {OUTPUT_IMG}")
