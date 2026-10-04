import os
import glob
import re
import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ============================================================
# RQ3 RESULT EXTRACTION
# ============================================================

ROOT = "rq3_relatedness"
OUTPUT_DIR = "rq3_analysis"

ENVIRONMENTS = [
    "hopper-friction",
    "halfcheetah-friction",
    "walker2d-friction",
    "ant-friction",
]

ALGORITHMS = [
    "BC_SAC",
    "RLPD",
    "BC_PAR",
]

SHIFTS = [
    "0.1",
    "0.5",
    "2.0",
    "5.0",
]

SEEDS = [
    "r0",
    "r1",
    "r2",
]


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# STORAGE
# ============================================================

learning_rows = []
diagnostic_rows = []
run_summary_rows = []


# ============================================================
# READ EACH EXPECTED RUN
# ============================================================

for environment in ENVIRONMENTS:

    for shift in SHIFTS:

        env_shift = (
            f"{environment}-srcdatatype-medium-{shift}"
        )

        for algorithm in ALGORITHMS:

            for seed in SEEDS:

                pattern = os.path.join(
                    ROOT,
                    algorithm,
                    env_shift,
                    seed,
                    "tb",
                    "events.out.tfevents.*",
                )

                files = glob.glob(pattern)

                if len(files) == 0:
                    print(
                        "MISSING:",
                        algorithm,
                        env_shift,
                        seed,
                    )
                    continue

                # ------------------------------------------------
                # A run may contain more than one TensorBoard file.
                # Read all of them.
                # ------------------------------------------------

                target_values = {}

                diagnostic_values = {
                    "train/distance": {},
                    "train/src_reward": {},
                    "train/encoder_loss": {},
                    "train/q1": {},
                    "train/logprob": {},
                }

                available_tags = set()

                for event_file in files:

                    try:
                        ea = EventAccumulator(event_file)
                        ea.Reload()

                        scalar_tags = ea.Tags()["scalars"]

                        available_tags.update(scalar_tags)

                        # ==========================================
                        # TARGET NORMALIZED SCORE
                        # ==========================================

                        target_tag = (
                            "test/target_normalized_score"
                        )

                        if target_tag in scalar_tags:

                            for event in ea.Scalars(target_tag):

                                target_values[event.step] = (
                                    event.value
                                )

                        # ==========================================
                        # DIAGNOSTICS
                        # ==========================================

                        for tag in diagnostic_values:

                            if tag in scalar_tags:

                                for event in ea.Scalars(tag):

                                    diagnostic_values[tag][
                                        event.step
                                    ] = event.value

                    except Exception as error:

                        print(
                            "ERROR READING:",
                            event_file,
                            error,
                        )

                # =================================================
                # SAVE LEARNING CURVE
                # =================================================

                for step in sorted(target_values):

                    learning_rows.append(
                        {
                            "environment": environment,
                            "shift": float(shift),
                            "algorithm": algorithm,
                            "seed": int(seed[1:]),
                            "step": int(step),
                            "target_normalized_score":
                                target_values[step],
                        }
                    )

                # =================================================
                # SAVE DIAGNOSTICS
                # =================================================

                for tag, values in diagnostic_values.items():

                    for step in sorted(values):

                        diagnostic_rows.append(
                            {
                                "environment": environment,
                                "shift": float(shift),
                                "algorithm": algorithm,
                                "seed": int(seed[1:]),
                                "step": int(step),
                                "metric": tag,
                                "value": values[step],
                            }
                        )

                # =================================================
                # RUN COMPLETENESS
                # =================================================

                if target_values:

                    max_step = max(target_values.keys())

                    final_score = target_values[max_step]

                else:

                    max_step = 0
                    final_score = float("nan")

                run_summary_rows.append(
                    {
                        "environment": environment,
                        "shift": float(shift),
                        "algorithm": algorithm,
                        "seed": int(seed[1:]),
                        "max_eval_step": max_step,
                        "final_score": final_score,
                        "num_eval_points":
                            len(target_values),
                    }
                )


# ============================================================
# DATAFRAMES
# ============================================================

learning_df = pd.DataFrame(learning_rows)

diagnostic_df = pd.DataFrame(diagnostic_rows)

run_summary_df = pd.DataFrame(run_summary_rows)


# ============================================================
# SORT
# ============================================================

if not learning_df.empty:

    learning_df = learning_df.sort_values(
        [
            "environment",
            "shift",
            "algorithm",
            "seed",
            "step",
        ]
    )


if not diagnostic_df.empty:

    diagnostic_df = diagnostic_df.sort_values(
        [
            "environment",
            "shift",
            "algorithm",
            "seed",
            "metric",
            "step",
        ]
    )


if not run_summary_df.empty:

    run_summary_df = run_summary_df.sort_values(
        [
            "environment",
            "shift",
            "algorithm",
            "seed",
        ]
    )


# ============================================================
# SAVE CSV FILES
# ============================================================

learning_file = os.path.join(
    OUTPUT_DIR,
    "rq3_learning_curves.csv",
)

diagnostic_file = os.path.join(
    OUTPUT_DIR,
    "rq3_diagnostics.csv",
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "rq3_run_summary.csv",
)


learning_df.to_csv(
    learning_file,
    index=False,
)

diagnostic_df.to_csv(
    diagnostic_file,
    index=False,
)

run_summary_df.to_csv(
    summary_file,
    index=False,
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 70)
print("RQ3 EXTRACTION SUMMARY")
print("=" * 70)

print("Expected runs: 144")
print("Runs found:", len(run_summary_df))

complete = (
    run_summary_df["max_eval_step"] >= 10000
).sum()

incomplete = (
    run_summary_df["max_eval_step"] < 10000
).sum()

print("Complete at 10000:", complete)
print("Incomplete:", incomplete)

print()
print(
    "Learning-curve rows:",
    len(learning_df),
)

print(
    "Diagnostic rows:",
    len(diagnostic_df),
)


# ============================================================
# COUNTS BY ENVIRONMENT
# ============================================================

print()
print("RUNS BY ENVIRONMENT")
print("-" * 70)

print(
    run_summary_df.groupby(
        "environment"
    ).size()
)


# ============================================================
# COUNTS BY ALGORITHM
# ============================================================

print()
print("RUNS BY ALGORITHM")
print("-" * 70)

print(
    run_summary_df.groupby(
        "algorithm"
    ).size()
)


# ============================================================
# EVALUATION POINT CHECK
# ============================================================

print()
print("EVALUATION POINT COUNTS")
print("-" * 70)

print(
    run_summary_df[
        "num_eval_points"
    ].value_counts().sort_index()
)


# ============================================================
# BC_PAR DIAGNOSTICS
# ============================================================

if not diagnostic_df.empty:

    bcpar_diag = diagnostic_df[
        diagnostic_df["algorithm"] == "BC_PAR"
    ]

    print()
    print("BC_PAR DIAGNOSTIC METRICS")
    print("-" * 70)

    print(
        bcpar_diag[
            "metric"
        ].value_counts()
    )


print()
print("Saved:")
print(learning_file)
print(diagnostic_file)
print(summary_file)

print()
print("RQ3 extraction finished.")