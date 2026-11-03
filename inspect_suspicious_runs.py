import pandas as pd

f = "matched_final_5000/matched_final_288_long.csv"
d = pd.read_csv(f)

checks = [
    ("hopper", 0.1),
    ("hopper", 0.5),
    ("walker2d", 0.1),
    ("walker2d", 0.5),
    ("walker2d", 2.0),
    ("walker2d", 5.0)
]

for env, shift in checks:
    print("\n" + "="*90)
    print(f"{env.upper()} | SHIFT {shift}")
    print("="*90)

    subset = d[
        (d["environment"] == env) &
        (d["shift"] == shift) &
        (d["method"].isin([
            "BC_SAC",
            "EA_NTC_NORMALIZED",
            "EA_NTC_UNNORMALIZED"
        ]))
    ]

    for _, r in subset.sort_values(
        ["seed", "method"]
    ).iterrows():

        print(
            f"\nMethod: {r['method']}"
            f"\nSeed: {r['seed']}"
            f"\nPath: {r['path']}"
            f"\nLog: {r['log_path']}"
            f"\nEvent hash: {r['event_hashes']}"
            f"\nConfig: {r['config_signature']}"
            f"\nEvaluations: "
            f"{[r[f'eval_{s}'] for s in [1000,2000,3000,4000,5000]]}"
        )
