import pandas as pd

df = pd.read_csv("full_5000_audit/all_candidates.csv")

print("\n========== AUDIT COLUMNS ==========")
print(df.columns.tolist())

for env in ["hopper", "halfcheetah", "walker2d", "ant"]:

    sub = df[df.environment == env]

    print(f"\n========== {env.upper()} ==========")

    for method in sorted(sub.method.unique()):

        x = sub[sub.method == method]

        print(f"\n{method}")
        print("Candidates:", len(x))
        print("Complete 5000:",
              x["complete_5000"].astype(str).str.lower().eq("true").sum())

        print("\nCoverage by shift and seed:")

        print(
            x.groupby("shift")["seed"]
             .apply(lambda s: sorted(set(s)))
             .to_string()
        )

        print("\nActual source directories:")
        print(x["path"].drop_duplicates().to_string(index=False))

