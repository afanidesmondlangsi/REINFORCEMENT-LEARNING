import pandas as pd
from pathlib import Path

AUDIT = Path("full_5000_audit/all_candidates.csv")

df = pd.read_csv(AUDIT)

methods = [
    "BC_PAR",
    "BC_SAC",
    "RLPD",
    "SAC_TARGET_ONLY",
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED"
]

df = df[
    df["method"].isin(methods)
    & df["seed"].isin([0, 1, 2])
    & df["complete_5000"].astype(str).str.lower().eq("true")
].copy()

keys = ["environment", "shift", "seed", "method"]

counts = df.groupby(keys).size().reset_index(name="count")

duplicates = counts[counts["count"] > 1]

detail = df.merge(
    duplicates[keys],
    on=keys,
    how="inner"
)

detail = detail.sort_values(keys + ["path"])

detail.to_csv(
    "full_5000_audit/duplicate_candidates_detail.csv",
    index=False
)

print("Duplicate configurations:", len(duplicates))
print("Duplicate candidate records:", len(detail))

for _, group in detail.groupby(keys):
    print("\n", group.iloc[0][keys].to_dict())

    for path in group["path"]:
        print("  ", path)

print("\nSaved duplicate_candidates_detail.csv")