from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os
import csv

runs = {
    ("SAC_TARGET_ONLY", 0.5):
        r"rq2_generalization\SAC_TARGET_ONLY\hopper-friction-0.5",

    ("SAC_TARGET_ONLY", 2.0):
        r"rq2_generalization\SAC_TARGET_ONLY\hopper-friction-2.0",

    ("BC_SAC", 0.5):
        r"rq2_generalization\BC_SAC\hopper-friction-srcdatatype-medium-0.5",

    ("BC_SAC", 2.0):
        r"rq2_generalization\BC_SAC\hopper-friction-srcdatatype-medium-2.0",

    ("RLPD", 0.5):
        r"rq2_generalization\RLPD\hopper-friction-srcdatatype-medium-0.5",

    ("RLPD", 2.0):
        r"rq2_generalization\RLPD\hopper-friction-srcdatatype-medium-2.0"
}

tag = "test/target_normalized_score"

rows = []

for (algorithm, shift), folder in runs.items():

    for seed in [0, 1, 2]:

        pattern = os.path.join(
            folder,
            f"r{seed}",
            "tb",
            "events.out.tfevents.*"
        )

        files = [
            f for f in glob.glob(pattern)
            if "incomplete" not in f.lower()
        ]

        if not files:
            print(f"MISSING: {algorithm}, shift={shift}, seed={seed}")
            continue

        event_file = max(files, key=os.path.getmtime)

        ea = EventAccumulator(event_file)
        ea.Reload()

        events = ea.Scalars(tag)

        for event in events:

            rows.append({
                "algorithm": algorithm,
                "shift": shift,
                "seed": seed,
                "step": event.step,
                "normalized_score": event.value
            })


# ------------------------------------------------------------
# Save complete results
# ------------------------------------------------------------

output_file = "rq2_all_results.csv"

with open(output_file, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "algorithm",
            "shift",
            "seed",
            "step",
            "normalized_score"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


print()
print("=" * 70)
print("RQ2 DATA EXTRACTION COMPLETE")
print("=" * 70)

print(f"Number of rows: {len(rows)}")
print(f"Saved to: {output_file}")

print("=" * 70)# ============================================================
# RQ2 SUMMARY STATISTICS
# Mean and standard deviation across the 3 seeds
# ============================================================

import pandas as pd

df = pd.read_csv("rq2_all_results.csv")

summary = (
    df.groupby(
        ["algorithm", "shift", "step"],
        as_index=False
    )
    .agg(
        mean_score=("normalized_score", "mean"),
        std_score=("normalized_score", "std"),
        min_score=("normalized_score", "min"),
        max_score=("normalized_score", "max"),
        n_seeds=("normalized_score", "count")
    )
)

summary.to_csv(
    "rq2_summary.csv",
    index=False
)

print()
print("=" * 70)
print("RQ2 SUMMARY STATISTICS")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("Saved to: rq2_summary.csv")


# ============================================================
# FINAL PERFORMANCE AT STEP 5000
# ============================================================

final_results = summary[
    summary["step"] == 5000
].copy()

print()
print("=" * 70)
print("FINAL PERFORMANCE AT 5000 TARGET STEPS")
print("=" * 70)

print(
    final_results[
        [
            "algorithm",
            "shift",
            "mean_score",
            "std_score",
            "min_score",
            "max_score"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

final_results.to_csv(
    "rq2_final_results.csv",
    index=False
)

print()
print("Saved to: rq2_final_results.csv")
# ============================================================
# RQ2 LEARNING CURVES
# Mean normalized score +/- 1 standard deviation
# ============================================================

import matplotlib.pyplot as plt

algorithms = ["SAC_TARGET_ONLY", "BC_SAC", "RLPD"]

for shift in [0.5, 2.0]:

    plt.figure(figsize=(9, 6))

    for algorithm in algorithms:

        temp = summary[
            (summary["algorithm"] == algorithm) &
            (summary["shift"] == shift)
        ].sort_values("step")

        x = temp["step"].to_numpy()
        mean = temp["mean_score"].to_numpy()
        std = temp["std_score"].to_numpy()

        plt.plot(
            x,
            mean,
            marker="o",
            linewidth=2,
            label=algorithm
        )

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.20
        )

    plt.xlabel("Target Environment Steps")
    plt.ylabel("Target Normalized Score")

    plt.title(
        f"RQ2: Hopper-Friction Generalization (Shift = {shift})"
    )

    plt.xticks(
        [1000, 2000, 3000, 4000, 5000]
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = f"rq2_learning_curve_shift_{shift}.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")

# ============================================================
# RQ2 AREA UNDER THE LEARNING CURVE
# Calculate AUC separately for every seed
# ============================================================

import numpy as np

auc_rows = []

for algorithm in df["algorithm"].unique():

    for shift in sorted(df["shift"].unique()):

        for seed in sorted(df["seed"].unique()):

            temp = df[
                (df["algorithm"] == algorithm) &
                (df["shift"] == shift) &
                (df["seed"] == seed)
            ].sort_values("step")

            if len(temp) != 5:
                continue

            x = temp["step"].to_numpy()
            y = temp["normalized_score"].to_numpy()

            # Trapezoidal area under learning curve
            auc = np.trapz(y, x)

            # Divide by total step interval (5000 - 1000)
            # so the result remains on the normalized-score scale
            average_performance = auc / (x[-1] - x[0])

            auc_rows.append({
                "algorithm": algorithm,
                "shift": shift,
                "seed": seed,
                "auc": auc,
                "average_performance": average_performance
            })


auc_df = pd.DataFrame(auc_rows)

auc_df.to_csv(
    "rq2_auc_by_seed.csv",
    index=False
)


# ============================================================
# Mean and SD across seeds
# ============================================================

auc_summary = (
    auc_df.groupby(
        ["algorithm", "shift"],
        as_index=False
    )
    .agg(
        mean_auc=("auc", "mean"),
        std_auc=("auc", "std"),
        mean_average_performance=("average_performance", "mean"),
        std_average_performance=("average_performance", "std")
    )
)

auc_summary.to_csv(
    "rq2_auc_summary.csv",
    index=False
)

print()
print("=" * 80)
print("RQ2 LEARNING-CURVE AUC")
print("=" * 80)

print(
    auc_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("Saved to: rq2_auc_by_seed.csv")
print("Saved to: rq2_auc_summary.csv")
# ============================================================
# RQ2 TRANSFER RELATIVE TO TARGET-ONLY SAC
# ============================================================

print()
print("=" * 85)
print("RQ2 TRANSFER RELATIVE TO SAC_TARGET_ONLY")
print("=" * 85)

transfer_rows = []

for shift in sorted(auc_summary["shift"].unique()):

    baseline = auc_summary[
        (auc_summary["algorithm"] == "SAC_TARGET_ONLY") &
        (auc_summary["shift"] == shift)
    ]["mean_average_performance"].iloc[0]

    for algorithm in ["BC_SAC", "RLPD"]:

        value = auc_summary[
            (auc_summary["algorithm"] == algorithm) &
            (auc_summary["shift"] == shift)
        ]["mean_average_performance"].iloc[0]

        absolute_difference = value - baseline

        percent_difference = (
            absolute_difference / baseline
        ) * 100

        transfer_rows.append({
            "algorithm": algorithm,
            "shift": shift,
            "average_performance": value,
            "target_only_baseline": baseline,
            "absolute_difference": absolute_difference,
            "percent_difference": percent_difference
        })


transfer_df = pd.DataFrame(transfer_rows)

print(
    transfer_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

transfer_df.to_csv(
    "rq2_transfer_vs_target_only.csv",
    index=False
)

print()
print("Saved to: rq2_transfer_vs_target_only.csv")
# ============================================================
# RQ2 SHIFT SENSITIVITY
# Compare average learning performance at shift 0.5 vs 2.0
# ============================================================

print()
print("=" * 85)
print("RQ2 SHIFT SENSITIVITY")
print("=" * 85)

shift_rows = []

for algorithm in ["SAC_TARGET_ONLY", "BC_SAC", "RLPD"]:

    score_05 = auc_summary[
        (auc_summary["algorithm"] == algorithm) &
        (auc_summary["shift"] == 0.5)
    ]["mean_average_performance"].iloc[0]

    score_20 = auc_summary[
        (auc_summary["algorithm"] == algorithm) &
        (auc_summary["shift"] == 2.0)
    ]["mean_average_performance"].iloc[0]

    absolute_change = score_20 - score_05

    percent_change = (
        absolute_change / score_05
    ) * 100

    shift_rows.append({
        "algorithm": algorithm,
        "shift_0.5_performance": score_05,
        "shift_2.0_performance": score_20,
        "absolute_change": absolute_change,
        "percent_change": percent_change
    })


shift_df = pd.DataFrame(shift_rows)

print(
    shift_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

shift_df.to_csv(
    "rq2_shift_sensitivity.csv",
    index=False
)

print()
print("Saved to: rq2_shift_sensitivity.csv")