import pandas as pd

p = "matched_final_5000/matched_final_288_long.csv"
d = pd.read_csv(p)

print("\n========== METRIC TAG DISTRIBUTION ==========")
print(d["tag"].value_counts(dropna=False).to_string())

print("\n========== ALGORITHM DISTRIBUTION ==========")
print(d["method"].value_counts().to_string())

print("\n========== REDUCTION DISTRIBUTION ==========")
print(d.groupby(["method", "reduction"]).size().to_string())

print("\n========== ENVIRONMENT COVERAGE ==========")
print(d.groupby("environment").size().to_string())

print("\n========== SHIFT COVERAGE ==========")
print(d.groupby(["environment", "shift"]).size().to_string())

print("\n========== SEED COVERAGE ==========")
print(sorted(d["seed"].unique()))

print("\n========== DUPLICATE CHECK ==========")
keys = ["environment", "shift", "seed", "method"]
print("Duplicate records:", d.duplicated(keys).sum())

print("\n========== EVALUATION VALUE CHECK ==========")
cols = ["eval_1000", "eval_2000", "eval_3000",
        "eval_4000", "eval_5000"]
print(d[cols].isna().sum().to_string())

print("\n========== EA DIAGNOSTIC TAGS ==========")
p = "matched_final_5000/ea_ntc_diagnostics/ea_ntc_diagnostics_timeseries.csv"
diag = pd.read_csv(p)
print(diag["tag"].value_counts().to_string())
