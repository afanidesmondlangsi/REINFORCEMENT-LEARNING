from pathlib import Path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ============================================================
# RQ1 HOPPER ANALYSIS
# ============================================================

TAG = "test/target_normalized_score"
REQUIRED_STEPS = [1000, 2000, 3000, 4000, 5000]

# Explicit locations verified by verify_rq1_hopper.py
RUNS = [
    # SAC_TARGET_ONLY
    (
        "SAC_TARGET_ONLY",
        0,
        Path(
            "rq1_target_only/SAC_TARGET_ONLY/"
            "hopper-friction-0.1/r0/tb"
        ),
    ),
    (
        "SAC_TARGET_ONLY",
        1,
        Path(
            "rq1_target_only/SAC_TARGET_ONLY/"
            "hopper-friction-0.1/r1/tb"
        ),
    ),
    (
        "SAC_TARGET_ONLY",
        2,
        Path(
            "rq1_target_only/SAC_TARGET_ONLY/"
            "hopper-friction-0.1/r2/tb"
        ),
    ),

    # BC_SAC
    (
        "BC_SAC",
        0,
        Path(
            "rq1_test/BC_SAC/"
            "hopper-friction-srcdatatype-medium-0.1/r0/tb"
        ),
    ),
    (
        "BC_SAC",
        1,
        Path(
            "rq1_bc_sac/BC_SAC/"
            "hopper-friction-srcdatatype-medium-0.1/r1/tb"
        ),
    ),
    (
        "BC_SAC",
        2,
        Path(
            "rq1_bc_sac/BC_SAC/"
            "hopper-friction-srcdatatype-medium-0.1/r2/tb"
        ),
    ),

    # RLPD
    (
        "RLPD",
        0,
        Path(
            "rq1_rlpd_test/RLPD/"
            "hopper-friction-srcdatatype-medium-0.1/r0/tb"
        ),
    ),
    (
        "RLPD",
        1,
        Path(
            "rq1_rlpd/RLPD/"
            "hopper-friction-srcdatatype-medium-0.1/r1/tb"
        ),
    ),
    (
        "RLPD",
        2,
        Path(
            "rq1_rlpd/RLPD/"
            "hopper-friction-srcdatatype-medium-0.1/r2/tb"
        ),
    ),
]


# ============================================================
# READ ONE COMPLETE RUN
# ============================================================

def read_run(algorithm, seed, tb_dir):

    event_files = list(tb_dir.glob("events.out.tfevents.*"))

    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event file found in {tb_dir}"
        )

    # Use newest event file if more than one exists
    event_file = max(
        event_files,
        key=lambda p: p.stat().st_mtime
    )

    ea = EventAccumulator(str(event_file))
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])

    if TAG not in scalar_tags:
        raise RuntimeError(
            f"{TAG} not found in {event_file}"
        )

    events = ea.Scalars(TAG)

    scores = {
        int(event.step): float(event.value)
        for event in events
    }

    missing = [
        step
        for step in REQUIRED_STEPS
        if step not in scores
    ]

    if missing:
        raise RuntimeError(
            f"{algorithm} seed {seed} is missing steps {missing}"
        )

    x = np.array(REQUIRED_STEPS, dtype=float)

    y = np.array(
        [scores[step] for step in REQUIRED_STEPS],
        dtype=float
    )

    # AUC over target interactions 1000 -> 5000
    auc = np.trapz(y, x)

    # Mean height of the learning curve over that interval
    average_performance = auc / (5000 - 1000)

    return {
        "environment": "hopper",
        "algorithm": algorithm,
        "seed": seed,
        "score_1000": scores[1000],
        "score_2000": scores[2000],
        "score_5000": scores[5000],
        "auc": auc,
        "average_performance": average_performance,
    }


# ============================================================
# PER-SEED RESULTS
# ============================================================

rows = []

for algorithm, seed, tb_dir in RUNS:
    rows.append(
        read_run(
            algorithm=algorithm,
            seed=seed,
            tb_dir=tb_dir,
        )
    )

by_seed = pd.DataFrame(rows)

by_seed = by_seed.sort_values(
    ["algorithm", "seed"]
).reset_index(drop=True)

print()
print("=" * 60)
print("RQ1 hopper - PER-SEED RESULTS")
print("=" * 60)
print()
print(by_seed.to_string(index=False))


# ============================================================
# SUMMARY ACROSS 3 SEEDS
# ============================================================

summary_rows = []

for algorithm, group in by_seed.groupby("algorithm"):

    summary_rows.append(
        {
            "algorithm": algorithm,
            "n": len(group),

            "score_1000_mean":
                group["score_1000"].mean(),

            "score_1000_sd":
                group["score_1000"].std(ddof=1),

            "score_2000_mean":
                group["score_2000"].mean(),

            "score_2000_sd":
                group["score_2000"].std(ddof=1),

            "final_mean":
                group["score_5000"].mean(),

            "final_sd":
                group["score_5000"].std(ddof=1),

            "auc_mean":
                group["auc"].mean(),

            "auc_sd":
                group["auc"].std(ddof=1),

            "average_performance_mean":
                group["average_performance"].mean(),

            "average_performance_sd":
                group["average_performance"].std(ddof=1),
        }
    )

summary = pd.DataFrame(summary_rows)

print()
print("=" * 60)
print("RQ1 hopper - SUMMARY ACROSS 3 SEEDS")
print("=" * 60)
print()
print(summary.to_string(index=False))


# ============================================================
# TRANSFER RELATIVE TO SAC_TARGET_ONLY
# ============================================================

baseline = summary[
    summary["algorithm"] == "SAC_TARGET_ONLY"
].iloc[0]

transfer_rows = []

for algorithm in ["BC_SAC", "RLPD"]:

    method = summary[
        summary["algorithm"] == algorithm
    ].iloc[0]

    transfer_rows.append(
        {
            "algorithm": algorithm,

            "delta_score_1000":
                method["score_1000_mean"]
                - baseline["score_1000_mean"],

            "delta_score_2000":
                method["score_2000_mean"]
                - baseline["score_2000_mean"],

            "delta_auc":
                method["auc_mean"]
                - baseline["auc_mean"],

            "delta_average_performance":
                method["average_performance_mean"]
                - baseline["average_performance_mean"],
        }
    )

transfer = pd.DataFrame(transfer_rows)

print()
print("=" * 60)
print("RQ1 hopper - TRANSFER VS SAC_TARGET_ONLY")
print("=" * 60)
print()
print(transfer.to_string(index=False))


# ============================================================
# SAVE RESULTS
# ============================================================

by_seed.to_csv(
    "rq1_hopper_by_seed.csv",
    index=False
)

summary.to_csv(
    "rq1_hopper_summary.csv",
    index=False
)

transfer.to_csv(
    "rq1_hopper_transfer.csv",
    index=False
)

print()
print("=" * 60)
print("FILES CREATED")
print("=" * 60)

print("rq1_hopper_by_seed.csv")
print("rq1_hopper_summary.csv")
print("rq1_hopper_transfer.csv")

print()
print("RQ1 hopper analysis complete.")