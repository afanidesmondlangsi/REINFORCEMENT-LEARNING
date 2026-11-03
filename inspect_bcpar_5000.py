import pandas as pd

df = pd.read_csv(
    "matched_5000_final/matched_5000_long.csv"
)

df = df[df.method == "BC_PAR"]

cols = [
    "environment", "shift", "seed",
    "config_signature", "normalized_score_1000",
    "normalized_score_2000", "normalized_score_3000",
    "normalized_score_4000", "normalized_score_5000",
    "target_return_5000"
]

print(df[cols].to_string(index=False))
