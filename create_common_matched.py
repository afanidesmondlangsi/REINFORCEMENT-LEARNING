import pandas as pd
from pathlib import Path

folder = Path("full_5000_audit")
coverage = pd.read_csv(folder / "coverage_576_slots.csv")

methods = [
    "SAC_TARGET_ONLY",
    "BC_SAC",
    "RLPD",
    "BC_PAR",
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED"
]

keys = ["environment", "shift", "seed"]

pivot = coverage.pivot(
    index=keys,
    columns="method",
    values="eligible_candidates"
).fillna(0)

available = pivot[(pivot[methods] >= 1).all(axis=1)]
unique = pivot[(pivot[methods] == 1).all(axis=1)]

print("\n========== MATCHING DIAGNOSTIC ==========")
print("Conditions with candidates for all six methods:", len(available))
print("Conditions with exactly one candidate per method:", len(unique))

print("\nCoverage by algorithm:")
for method in methods:
    print(
        f"{method:24s}: "
        f"{int((pivot[method] >= 1).sum())} available, "
        f"{int((pivot[method] > 1).sum())} duplicates"
    )

print("\nCoverage by seed:")
for seed in range(6):
    print(f"Seed {seed}: {(available.reset_index()['seed'] == seed).sum()}")

print("\nCoverage by environment:")
print(available.reset_index().groupby("environment").size())

print("\nCoverage by shift:")
print(available.reset_index().groupby("shift").size())

available.reset_index().to_csv(
    folder / "matched_candidates_for_review.csv",
    index=False
)

unique.reset_index().to_csv(
    folder / "strict_unique_matched.csv",
    index=False
)

print("\nReports saved successfully in:", folder.resolve())
