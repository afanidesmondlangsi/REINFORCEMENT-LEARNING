from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("analysis_correct_runs_4env")
paired = pd.read_csv(ROOT / "paired_results.csv")
curves = pd.read_csv(ROOT / "learning_curves.csv")
audit = pd.read_csv(ROOT / "run_audit.csv")

OUT = ROOT / "failure_diagnostics"
OUT.mkdir(exist_ok=True)

# -------------------------------------------------------
# 1. FIRST DIVERGENCE CHECK
# -------------------------------------------------------

wide = curves.pivot_table(
    index=["environment", "shift", "seed", "step"],
    columns="variant",
    values="score",
    aggfunc="first"
).reset_index()

wide["difference"] = (
    wide["NORMALIZED"] - wide["UNNORMALIZED"]
)

wide["different"] = ~np.isclose(
    wide["NORMALIZED"],
    wide["UNNORMALIZED"],
    rtol=1e-7,
    atol=1e-7
)

divergence = (
    wide[wide["different"]]
    .groupby(["environment", "shift", "seed"])["step"]
    .min()
    .rename("first_divergence_step")
    .reset_index()
)

all_pairs = (
    wide[["environment", "shift", "seed"]]
    .drop_duplicates()
)

divergence = all_pairs.merge(
    divergence,
    on=["environment", "shift", "seed"],
    how="left"
)

divergence.to_csv(
    OUT / "first_divergence.csv",
    index=False
)

print("\n========== FIRST DIVERGENCE ==========")
print(
    divergence.groupby(
        ["environment", "shift"],
        dropna=False
    )["first_divergence_step"]
    .agg(["count", "min", "median", "max"])
    .to_string()
)

print("\nEARLY CHECKPOINT EXACT EQUALITY")
for step in [1000, 2000, 3000, 4000, 5000]:
    subset = wide[wide.step == step]
    equal = (~subset["different"]).sum()
    print(
        f"Step {step}: {equal}/{len(subset)} paired scores equal"
    )

# -------------------------------------------------------
# 2. WALKER2D SHIFT 2.0 FAILURE
# -------------------------------------------------------

walker = paired[
    (paired.environment == "walker2d") &
    (np.isclose(paired["shift"], 2.0))
].copy()

columns = [
    "seed",
    "auc_norm",
    "auc_mean",
    "delta_auc",
    "final_score_norm",
    "final_score_mean",
    "delta_final_score"
]

print("\n========== WALKER2D SHIFT 2.0 ==========")
print(
    walker[columns]
    .sort_values("delta_auc")
    .to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

walker.to_csv(
    OUT / "walker2d_shift_2_failure.csv",
    index=False
)

# -------------------------------------------------------
# 3. CORRECTED HOPPER FAILURE CONDITIONS
# -------------------------------------------------------

hopper = paired[
    paired.environment == "hopper"
].copy()

print("\n========== CORRECTED HOPPER ==========")
print(
    hopper.groupby("shift")
    .agg(
        mean_auc_difference=("delta_auc", "mean"),
        median_auc_difference=("delta_auc", "median"),
        mean_final_difference=("delta_final_score", "mean"),
        negative_auc_seeds=(
            "delta_auc",
            lambda x: int((x < 0).sum())
        )
    )
    .to_string(float_format=lambda x: f"{x:.4f}")
)

# -------------------------------------------------------
# 4. GATE DIAGNOSTICS BY ENVIRONMENT AND VARIANT
# -------------------------------------------------------

diagnostic_columns = [
    "gate_ready",
    "gate_strength",
    "source_weight",
    "relative_mse"
]

available = [
    x for x in diagnostic_columns if x in audit.columns
]

print("\n========== GATE DIAGNOSTICS BY VARIANT ==========")

if available:
    print(
        audit.groupby(["environment", "variant"])[available]
        .mean()
        .to_string(float_format=lambda x: f"{x:.5f}")
    )

# -------------------------------------------------------
# 5. PAIRED GATE DIFFERENCES AND AUC
# -------------------------------------------------------

gate_pairs = paired.copy()

for diagnostic in available:
    a = diagnostic + "_norm"
    b = diagnostic + "_mean"

    if a in gate_pairs and b in gate_pairs:
        gate_pairs["delta_" + diagnostic] = (
            gate_pairs[a] - gate_pairs[b]
        )

print("\n========== AUC / GATE CORRELATIONS ==========")

for environment, subset in gate_pairs.groupby("environment"):
    print("\nEnvironment:", environment)

    for diagnostic in available:
        col = "delta_" + diagnostic

        if col not in subset:
            continue

        clean = subset[
            ["delta_auc", col]
        ].dropna()

        if (
            len(clean) >= 4
            and clean["delta_auc"].std() > 0
            and clean[col].std() > 0
        ):
            correlation = clean["delta_auc"].corr(clean[col])
            print(
                diagnostic,
                "paired correlation:",
                round(correlation, 4)
            )
        else:
            print(diagnostic, "insufficient variation")

gate_pairs.to_csv(
    OUT / "paired_gate_diagnostics.csv",
    index=False
)

print("\nDiagnostics saved to:", OUT)
