import pandas as pd

df = pd.read_csv("full_5000_audit/all_candidates.csv")

for env in ["hopper", "walker2d"]:
    print("\n==========", env.upper(), "==========")

    subset = df[df.environment == env]

    for method in sorted(subset.method.unique()):
        rows = subset[subset.method == method]

        print("\n", method)
        print(rows[["shift", "seed", "path", "complete_5000"]]
              .to_string(index=False))
