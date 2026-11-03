import pandas as pd

f = "matched_final_5000/matched_final_288_long.csv"
d = pd.read_csv(f)

ea = d[d["method"].isin([
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED"
])].copy()

metrics = [
    "eval_1000",
    "eval_2000",
    "eval_3000",
    "eval_4000",
    "eval_5000"
]

print("\n========== EA-NTC PAIRING AUDIT ==========")

for env in sorted(ea.environment.unique()):
    for shift in sorted(ea.loc[
        ea.environment == env, "shift"
    ].unique()):

        a = ea[
            (ea.environment == env) &
            (ea["shift"] == shift)
        ]

        wide = a.pivot(
            index="seed",
            columns="method",
            values=metrics
        )

        print(f"\n{env.upper()} | SHIFT {shift}")

        for metric in metrics:
            x = wide[metric]["EA_NTC_NORMALIZED"]
            y = wide[metric]["EA_NTC_UNNORMALIZED"]

            diff = x - y

            print(
                f"{metric}: "
                f"normalized={x.mean():.4f}, "
                f"mean={y.mean():.4f}, "
                f"difference={diff.mean():.4f}, "
                f"identical={x.equals(y)}"
            )

print("\n========== DUPLICATE PERFORMANCE AUDIT ==========")

for env in ["hopper", "walker2d"]:
    print(f"\n{env.upper()}")

    subset = ea[ea.environment == env]

    print(
        subset.groupby(
            ["shift", "method"]
        )[metrics].mean().round(5).to_string()
    )

print("\nAUDIT COMPLETE")
