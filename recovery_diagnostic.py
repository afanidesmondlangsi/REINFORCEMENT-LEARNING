import pandas as pd
from pathlib import Path

ROOT = Path("full_5000_audit")

candidates = pd.read_csv(ROOT / "all_candidates.csv")

print("\n========== EXISTING RUN COUNTS ==========")

print(
    candidates.groupby(["environment", "method"])
    .agg(
        candidates=("seed", "size"),
        unique_seeds=("seed", "nunique"),
        complete_5000=("complete_5000", "sum"),
        raw_returns=("raw_return_available", "sum")
    ).to_string()
)

print("\n========== RLPD AND SAC TARGET ONLY ==========")

for method in ["RLPD", "SAC_TARGET_ONLY"]:
    print(f"\n{method}")

    subset = candidates[candidates.method == method]

    if subset.empty:
        print("No recognized candidates.")
        continue

    print(
        subset[
            ["environment", "shift", "seed",
             "complete_5000", "raw_return_available", "path"]
        ].to_string(index=False)
    )

print("\n========== EA-NTC CLASSIFICATION ==========")

for method in ["EA_NTC_NORMALIZED", "EA_NTC_UNNORMALIZED"]:
    print(f"\n{method}")

    subset = candidates[candidates.method == method]

    print(
        subset.groupby(["environment", "shift"])
        .agg(
            runs=("seed", "size"),
            seeds=("seed", "nunique"),
            complete=("complete_5000", "sum")
        ).to_string()
    )

print("\n========== UNCLASSIFIED RUNS ==========")

file = ROOT / "unclassified_runs.csv"

if file.exists():
    unclassified = pd.read_csv(file)

    if len(unclassified):
        print(
            unclassified[
                ["environment", "shift", "seed", "method", "path"]
            ].to_string(index=False)
        )
    else:
        print("None")
else:
    print("Unclassified report was not generated.")

print("\n========== SCAN ERRORS ==========")

file = ROOT / "scan_errors.csv"

if file.exists():
    errors = pd.read_csv(file)

    print("Total scan errors:", len(errors))

    if len(errors):
        print(errors.to_string(index=False))
else:
    print("Scan error report unavailable.")

print("\n========== END OF RECOVERY DIAGNOSTIC ==========")
