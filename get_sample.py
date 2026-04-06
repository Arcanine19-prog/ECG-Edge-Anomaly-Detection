import pandas as pd
import numpy as np
import json
import sys
import os

with open("models/feature_indices.json") as f:
    meta = json.load(f)

indices = meta["top_15_indices"]
labels  = meta["feature_labels"]

csv_path = "mitbih_test.csv"
if not os.path.exists(csv_path):
    csv_path = "data/mitbih_test.csv"

print(f"Loading {csv_path}...")
data = pd.read_csv(csv_path, header=None)
X    = data.iloc[:, :-1].values
y    = data.iloc[:, -1].values

arg = sys.argv[1] if len(sys.argv) > 1 else "normal"

if arg == "normal":
    row_idx = np.where(y == 0)[0][0]
elif arg == "abnormal":
    row_idx = np.where(y != 0)[0][0]
else:
    row_idx = int(arg)

row      = X[row_idx]
top_vals = row[indices]
label    = int(y[row_idx])

print(f"\nRow {row_idx} — label: {'NORMAL' if label == 0 else 'ABNORMAL class ' + str(label)}")
print(f"\n{'Feature':<10} {'Value':>10}")
print("-" * 22)
for feat, val in zip(labels, top_vals):
    print(f"{feat:<10} {val:>10.4f}")

print(f"\nExpected prediction: {'Normal' if label == 0 else 'Anomaly'}")
print(f"\nComma separated values:")
print(", ".join(f"{v:.4f}" for v in top_vals))

normal_rows = X[y == 0][:, indices]
print(f"\nNormal value ranges from dataset:")
print(f"{'Feature':<10} {'Min':>8} {'Max':>8} {'Mean':>8}")
print("-" * 38)
for i, feat in enumerate(labels):
    col = normal_rows[:, i]
    print(f"{feat:<10} {col.min():>8.4f} {col.max():>8.4f} {col.mean():>8.4f}")