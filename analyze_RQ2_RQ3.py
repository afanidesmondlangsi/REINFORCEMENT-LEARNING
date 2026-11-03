import pandas as pd
import numpy as np
from pathlib import Path

root = Path("matched_final_5000")
out = root / "final_three_RQ_analysis"
tables = out / "paper_tables"

d = pd.read_csv(root / "matched_final_288_long.csv")

metrics = [
    "eval_1000", "eval_2000", "eval_3000",
    "eval_4000", "eval_5000"
]

# ------------------------------------------------------------
# RQ2: Performance by environment and friction shift
# ------------------------------------------------------------

print("\n========== RQ2: PERFORMANCE BY FRICTION SHIFT ==========")

summary = (
    d.groupby(["environment", "shift", "method"])[metrics]
    .agg(["mean", "std"])
)

summary.to_csv(tables / "RQ2_complete_shift_analysis.csv")

for env in sorted(d.environment.unique()):
    print(f"\n========== {env.upper()} ==========")

    for shift in sorted(d.loc[d['environment'] == env, 'shift'].unique()):
        print(f"\nSHIFT {shift}")

        sub = d[
            (d.environment == env) &
            (d['shift'] == shift)
        ]

        result = (
            sub.groupby("method")["eval_5000"]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=False)
        )

        print(result.round(3).to_string())

# ------------------------------------------------------------
# Paired EA-NTC comparisons by environment and shift
# ------------------------------------------------------------

print("\n========== EA NORMALIZED VS MEAN ==========")

norm = d[d.method == "EA_NTC_NORMALIZED"]
mean = d[d.method == "EA_NTC_UNNORMALIZED"]

keys = ["environment", "shift", "seed"]

paired = norm[keys + metrics].merge(
    mean[keys + metrics],
    on=keys,
    suffixes=("_normalized", "_mean"),
    validate="one_to_one"
)

for col in metrics:
    paired[col + "_difference"] = (
        paired[col + "_normalized"] -
        paired[col + "_mean"]
    )

paired.to_csv(
    tables / "RQ3_EA_paired_checkpoint_differences.csv",
    index=False
)

print(
    paired.groupby(["environment", "shift"])[
        ["eval_1000_difference",
         "eval_3000_difference",
         "eval_5000_difference"]
    ].mean().round(4).to_string()
)

# ------------------------------------------------------------
# Investigate zero Walker2d difference
# ------------------------------------------------------------

print("\n========== WALKER2D EQUALITY AUDIT ==========")

w = paired[paired.environment == "walker2d"].copy()

for col in [x + "_difference" for x in metrics]:
    print(
        col,
        "identical:",
        np.allclose(w[col], 0, atol=1e-10),
        "maximum absolute difference:",
        w[col].abs().max()
    )

# Compare recorded configurations and event hashes
for column in ["config_signature", "event_hashes"]:
    print(f"\n{column} comparison:")

    n = norm[norm.environment == "walker2d"][
        keys + [column]
    ]

    m = mean[mean.environment == "walker2d"][
        keys + [column]
    ]

    check = n.merge(
        m,
        on=keys,
        suffixes=("_normalized", "_mean")
    )

    print(
        "Identical pairs:",
        (
            check[column + "_normalized"] ==
            check[column + "_mean"]
        ).sum(),
        "of", len(check)
    )

# ------------------------------------------------------------
# RQ3: Diagnostic availability and behavior
# ------------------------------------------------------------

print("\n========== RQ3: ADAPTATION DIAGNOSTICS ==========")

diag = pd.read_csv(
    root / "ea_ntc_diagnostics" /
    "ea_ntc_diagnostics_timeseries.csv"
)

tags = [
    "ea_ntc/adaptive_gate_strength",
    "ea_ntc/mean_source_weight",
    "ea_ntc/source_weight_ess",
    "ea_ntc/source_weight_below_0_5",
    "ea_ntc/gate_ready",
    "ea_ntc/heldout_relative_mse",
    "ea_ntc/heldout_target_mse",
    "ea_ntc/heldout_zero_delta_mse"
]

diag = diag[diag.tag.isin(tags)].copy()

diag_summary = (
    diag.groupby(
        ["method", "environment", "shift", "tag"]
    )
    .agg(
        mean_value=("value", "mean"),
        std_value=("value", "std"),
        observations=("value", "count")
    )
    .reset_index()
)

diag_summary.to_csv(
    tables / "RQ3_full_adaptation_diagnostics.csv",
    index=False
)

print(
    diag_summary.groupby(["method", "tag"])[
        ["mean_value", "std_value"]
    ].mean().round(4).to_string()
)

print("\n========== ADDITIONAL ANALYSIS SAVED ==========")
print("RQ2:", tables / "RQ2_complete_shift_analysis.csv")
print("RQ3:", tables / "RQ3_full_adaptation_diagnostics.csv")
print("EA pairing:", tables / "RQ3_EA_paired_checkpoint_differences.csv")

