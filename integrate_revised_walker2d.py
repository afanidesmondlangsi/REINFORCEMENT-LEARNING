import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path("matched_final_5000")

MAIN = BASE / "matched_final_288_long.csv"

SOURCE = (
    BASE / "walker2d_revised_validation" /
    "walker2d_revised_24_records.csv"
)

METRICS = [
    "eval_1000",
    "eval_2000",
    "eval_3000",
    "eval_4000",
    "eval_5000"
]

print("\n========== WALKER2D INTEGRATION ==========")

# Load datasets
d = pd.read_csv(MAIN)
w = pd.read_csv(SOURCE)

# Map treatment names
method_map = {
    "NORMALIZED": "EA_NTC_NORMALIZED",
    "UNNORMALIZED": "EA_NTC_UNNORMALIZED"
}

w["method"] = w["treatment"].map(method_map)

# Validate source
assert len(w) == 24, f"Expected 24 records, found {len(w)}"
assert w["method"].notna().all(), "Unknown treatment found"
assert set(w["seed"]) == {0, 1, 2}
assert set(w["shift"]) == {0.1, 0.5, 2.0, 5.0}
assert (w["environment"] == "walker2d").all()
assert w[METRICS].notna().all().all()

# Matching keys
keys = ["method", "environment", "shift", "seed"]

assert not w.duplicated(keys).any(), "Duplicate source records"

# Identify existing Walker2d EA records
mask = (
    (d["environment"] == "walker2d") &
    (d["method"].isin(method_map.values()))
)

old = d.loc[mask].copy()

assert len(old) == 24, (
    f"Expected 24 existing Walker2d EA records, found {len(old)}"
)

assert not old.duplicated(keys).any(), "Duplicate destination records"

# Check exact one-to-one matching
check = old[keys].merge(
    w[keys],
    on=keys,
    how="outer",
    indicator=True,
    validate="one_to_one"
)

assert (check["_merge"] == "both").all(), (
    "Source and destination conditions do not match exactly"
)

# Backup original dataset
backup = BASE / (
    "matched_final_288_long_before_walker2d_revision_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".csv"
)

d.to_csv(backup, index=False)

# Prepare source lookup
w = w.set_index(keys)

# Update only the 24 matching records
for idx in d.index[mask]:

    key = tuple(d.loc[idx, k] for k in keys)

    source = w.loc[key]

    for metric in METRICS:
        d.at[idx, metric] = float(source[metric])

    d.at[idx, "path"] = str(source["path"]).replace("/", "\\")

    d.at[idx, "reduction"] = str(source["reduction"])

    if "path_norm" in d.columns:
        d.at[idx, "path_norm"] = (
            str(source["path"]).replace("/", "\\").lower()
        )

# Validate final dataset
assert len(d) == 288
assert d[METRICS].notna().all().all()

updated = d.loc[mask]

assert len(updated) == 24

# Save updated dataset
d.to_csv(MAIN, index=False)

print("\n========== INTEGRATION SUMMARY ==========")
print("Original records:", len(d))
print("Updated Walker2d records:", len(updated))
print("Unchanged records:", len(d) - len(updated))
print("Total records:", len(d))
print("Backup:", backup.resolve())
print("Updated dataset:", MAIN.resolve())

print("\n========== REVISED WALKER2D MEANS ==========")

print(
    updated.groupby(["shift", "method"])[METRICS]
    .mean()
    .round(3)
    .to_string()
)

print("\nSUCCESS: Revised Walker2d results integrated.")
