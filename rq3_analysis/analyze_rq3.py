import os
import numpy as np
import pandas as pd

# ============================================================
# RQ3 STATISTICAL ANALYSIS
# ============================================================

INPUT = os.path.join(
    "rq3_analysis",
    "rq3_learning_curves.csv"
)

OUTPUT_DIR = "rq3_analysis"

df = pd.read_csv(INPUT)

print("=" * 75)
print("RQ3 STATISTICAL ANALYSIS")
print("=" * 75)

print("Rows:", len(df))
print("Runs:", df[
    ["environment", "shift", "algorithm", "seed"]
].drop_duplicates().shape[0])

print("Steps:", sorted(df["step"].unique()))


# ============================================================
# 1. FINAL PERFORMANCE AT STEP 10000
# ============================================================

final_df = df[
    df["step"] == 10000
].copy()

final_summary = (
    final_df
    .groupby(
        ["environment", "shift", "algorithm"]
    )["target_normalized_score"]
    .agg(
        mean="mean",
        sd="std",
        median="median",
        min="min",
        max="max",
        n="count"
    )
    .reset_index()
)

final_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rq3_final_performance.csv"
    ),
    index=False
)


# ============================================================
# 2. POST-ADAPTATION PERFORMANCE
#
# Use evaluations from step 5000 through 10000.
# This summarizes sustained performance rather than relying
# only on the final evaluation.
# ============================================================

post_df = df[
    (df["step"] >= 5000) &
    (df["step"] <= 10000)
].copy()

post_run = (
    post_df
    .groupby(
        ["environment", "shift", "algorithm", "seed"]
    )["target_normalized_score"]
    .mean()
    .reset_index(
        name="post_adaptation_mean"
    )
)

post_summary = (
    post_run
    .groupby(
        ["environment", "shift", "algorithm"]
    )["post_adaptation_mean"]
    .agg(
        mean="mean",
        sd="std",
        median="median",
        n="count"
    )
    .reset_index()
)

post_run.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rq3_post_adaptation_by_run.csv"
    ),
    index=False
)

post_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rq3_post_adaptation_summary.csv"
    ),
    index=False
)


# ============================================================
# 3. AUC FROM STEP 5000 TO 10000
#
# Trapezoidal AUC.
# Divide by time span so values remain on approximately the
# same scale as normalized score.
# ============================================================

auc_rows = []

for keys, group in post_df.groupby(
    ["environment", "shift", "algorithm", "seed"]
):

    group = group.sort_values("step")

    x = group["step"].to_numpy()
    y = group[
        "target_normalized_score"
    ].to_numpy()

    if len(x) >= 2:

        auc = np.trapz(y, x)

        span = x[-1] - x[0]

        normalized_auc = (
            auc / span
            if span > 0
            else np.nan
        )

    else:

        auc = np.nan
        normalized_auc = np.nan

    auc_rows.append(
        {
            "environment": keys[0],
            "shift": keys[1],
            "algorithm": keys[2],
            "seed": keys[3],
            "auc_5000_10000": auc,
            "normalized_auc_5000_10000":
                normalized_auc,
        }
    )


auc_df = pd.DataFrame(auc_rows)

auc_summary = (
    auc_df
    .groupby(
        ["environment", "shift", "algorithm"]
    )["normalized_auc_5000_10000"]
    .agg(
        mean="mean",
        sd="std",
        median="median",
        n="count"
    )
    .reset_index()
)

auc_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rq3_auc_by_run.csv"
    ),
    index=False
)

auc_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rq3_auc_summary.csv"
    ),
    index=False
)


# ============================================================
# 4. BC_PAR IMPROVEMENT RELATIVE TO BASELINES
#
# Compare group means for the same environment and shift.
# ============================================================

comparison_rows = []

for (environment, shift), group in post_summary.groupby(
    ["environment", "shift"]
):

    values = dict(
        zip(
            group["algorithm"],
            group["mean"]
        )
    )

    if (
        "BC_PAR" in values
        and "BC_SAC" in values
        and "RLPD" in values
    ):

        bcpar = values["BC_PAR"]
        bcsac = values["BC_SAC"]
        rlpd = values["RLPD"]

        comparison_rows.append(
            {
                "environment": environment,
                "shift": shift,

                "BC_PAR":
                    bcpar,

                "BC_SAC":
                    bcsac,

                "RLPD":
                    rlpd,

                "BC_PAR_minus_BC_SAC":
                    bcpar - bcsac,

                "BC_PAR_minus_RLPD":
                    bcpar - rlpd,

                "best_algorithm":
                    max(
                        values,
                        key=values.get
                    )
            }
        )


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rq3_algorithm_comparison.csv"
    ),
    index=False
)


# ============================================================
# 5. WIN COUNTS
# ============================================================

win_counts = (
    comparison_df[
        "best_algorithm"
    ]
    .value_counts()
    .rename_axis("algorithm")
    .reset_index(name="wins")
)

win_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "rq3_win_counts.csv"
    ),
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 75)
print("FINAL PERFORMANCE AT STEP 10000")
print("=" * 75)

print(
    final_summary.to_string(
        index=False
    )
)


print()
print("=" * 75)
print("POST-ADAPTATION MEAN: STEPS 5000-10000")
print("=" * 75)

print(
    post_summary.to_string(
        index=False
    )
)


print()
print("=" * 75)
print("BC_PAR COMPARISON")
print("=" * 75)

print(
    comparison_df.to_string(
        index=False
    )
)


print()
print("=" * 75)
print("NUMBER OF ENVIRONMENT/SHIFT CONDITIONS WON")
print("=" * 75)

print(
    win_counts.to_string(
        index=False
    )
)


print()
print("Saved statistical summaries in rq3_analysis/")