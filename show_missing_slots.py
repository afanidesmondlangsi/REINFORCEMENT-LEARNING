import pandas as pd
from pathlib import Path

file = Path("full_5000_audit/coverage_576_slots.csv")
df = pd.read_csv(file)

for method in ["SAC_TARGET_ONLY", "RLPD"]:
    print("\n" + "=" * 65)
    print("MISSING CONDITIONS:", method)
    print("=" * 65)

    missing = df[
        (df["method"] == method) &
        (df["eligible_candidates"] == 0)
    ]

    print("Missing slots:", len(missing))

    for env in ["hopper", "halfcheetah", "walker2d", "ant"]:
        subset = missing[missing["environment"] == env]

        print(f"\n{env.upper()}: {len(subset)} missing")

        for shift in sorted(subset["shift"].unique()):
            seeds = sorted(
                subset[subset["shift"] == shift]["seed"].tolist()
            )
            print(f"  Shift {shift}: seeds {seeds}")

