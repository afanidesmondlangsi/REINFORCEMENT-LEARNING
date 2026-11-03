import pandas as pd

df = pd.read_csv(
    "matched_5000_final/matched_5000_long.csv"
)

metrics = [
    "normalized_score_5000",
    "target_return_5000",
    "auc_1000_5000",
]

for metric in metrics:

    print("\n" + "=" * 85)
    print("METRIC:", metric)
    print("=" * 85)

    result = df.groupby(
        ["environment", "shift", "method"]
    )[metric].agg(
        mean="mean",
        std="std",
        seeds="count"
    ).reset_index()

    result["mean_std"] = result.apply(
        lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}",
        axis=1
    )

    print(
        result[
            ["environment", "shift", "method",
             "mean_std", "seeds"]
        ].to_string(index=False)
    )

