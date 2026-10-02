from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import pandas as pd

ROOT = Path("rq1_generalization")
ENVIRONMENT = "walker2d-friction"
TAG = "test/target_normalized_score"

REQUIRED_STEPS = [1000, 2000, 3000, 4000, 5000]

ALGORITHMS = [
    "SAC_TARGET_ONLY",
    "BC_SAC",
    "RLPD"
]


def get_complete_run(tb_dir):

    event_files = sorted(
        tb_dir.glob("events.out.tfevents.*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    for event_file in event_files:

        try:
            ea = EventAccumulator(str(event_file))
            ea.Reload()

            if TAG not in ea.Tags().get("scalars", []):
                continue

            events = ea.Scalars(TAG)

            values = {
                int(e.step): float(e.value)
                for e in events
            }

            if all(step in values for step in REQUIRED_STEPS):

                scores = [
                    values[step]
                    for step in REQUIRED_STEPS
                ]

                return event_file, scores

        except Exception:
            continue

    return None, None


rows = []

for algorithm in ALGORITHMS:

    algorithm_dir = ROOT / algorithm

    if not algorithm_dir.exists():
        continue

    for tb_dir in algorithm_dir.rglob("tb"):

        path_text = str(tb_dir).lower()

        # Only walker2d
        if ENVIRONMENT not in path_text:
            continue

        # Only RQ1 shift = 0.1
        if "0.1" not in path_text:
            continue

        event_file, scores = get_complete_run(tb_dir)

        if scores is None:
            continue

        # Extract seed from r0, r1, r2
        seed = None

        for part in tb_dir.parts:
            if part in ["r0", "r1", "r2"]:
                seed = int(part[1:])

        if seed is None:
            continue

        steps = np.array(REQUIRED_STEPS, dtype=float)
        scores_array = np.array(scores, dtype=float)

        # Area under learning curve from 1000 to 5000
        auc = np.trapz(scores_array, steps)

        # Average performance over this interval
        average_performance = auc / (
            REQUIRED_STEPS[-1] - REQUIRED_STEPS[0]
        )

        rows.append({
            "environment": "walker2d",
            "algorithm": algorithm,
            "shift": 0.1,
            "seed": seed,

            "score_1000": scores[0],
            "score_2000": scores[1],
            "score_3000": scores[2],
            "score_4000": scores[3],
            "score_5000": scores[4],

            "auc": auc,
            "average_performance": average_performance,

            "event_file": event_file.name
        })


df = pd.DataFrame(rows)

df = df.sort_values(
    ["algorithm", "seed"]
).reset_index(drop=True)


print("\n============================================================")
print("RQ1 walker2d - PER-SEED RESULTS")
print("============================================================\n")

print(
    df[
        [
            "algorithm",
            "seed",
            "score_1000",
            "score_2000",
            "score_5000",
            "auc",
            "average_performance"
        ]
    ].to_string(index=False)
)


# ============================================================
# AGGREGATE ACROSS SEEDS
# ============================================================

summary = (
    df.groupby("algorithm")
    .agg(
        n=("seed", "count"),

        score_1000_mean=("score_1000", "mean"),
        score_1000_sd=("score_1000", "std"),

        score_2000_mean=("score_2000", "mean"),
        score_2000_sd=("score_2000", "std"),

        final_mean=("score_5000", "mean"),
        final_sd=("score_5000", "std"),

        auc_mean=("auc", "mean"),
        auc_sd=("auc", "std"),

        average_performance_mean=("average_performance", "mean"),
        average_performance_sd=("average_performance", "std")
    )
    .reset_index()
)


print("\n============================================================")
print("RQ1 walker2d - SUMMARY ACROSS 3 SEEDS")
print("============================================================\n")

print(summary.to_string(index=False))


# ============================================================
# TRANSFER RELATIVE TO SAC_TARGET_ONLY
# ============================================================

baseline_row = summary[
    summary["algorithm"] == "SAC_TARGET_ONLY"
].iloc[0]

baseline_auc = baseline_row["auc_mean"]
baseline_average = baseline_row["average_performance_mean"]
baseline_1000 = baseline_row["score_1000_mean"]
baseline_2000 = baseline_row["score_2000_mean"]


transfer_rows = []

for _, row in summary.iterrows():

    if row["algorithm"] == "SAC_TARGET_ONLY":
        continue

    transfer_rows.append({
        "algorithm": row["algorithm"],

        "delta_score_1000":
            row["score_1000_mean"] - baseline_1000,

        "delta_score_2000":
            row["score_2000_mean"] - baseline_2000,

        "delta_auc":
            row["auc_mean"] - baseline_auc,

        "delta_average_performance":
            row["average_performance_mean"] - baseline_average
    })


transfer = pd.DataFrame(transfer_rows)


print("\n============================================================")
print("RQ1 walker2d - TRANSFER VS SAC_TARGET_ONLY")
print("============================================================\n")

print(transfer.to_string(index=False))


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    "rq1_walker2d_by_seed.csv",
    index=False
)

summary.to_csv(
    "rq1_walker2d_summary.csv",
    index=False
)

transfer.to_csv(
    "rq1_walker2d_transfer.csv",
    index=False
)


print("\n============================================================")
print("FILES CREATED")
print("============================================================")

print("rq1_walker2d_by_seed.csv")
print("rq1_walker2d_summary.csv")
print("rq1_walker2d_transfer.csv")

print("\nRQ1 walker2d analysis complete.")
