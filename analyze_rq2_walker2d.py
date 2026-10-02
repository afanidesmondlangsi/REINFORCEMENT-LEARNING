from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path("rq2_generalization")
TAG = "test/target_normalized_score"

TARGET_STEPS = [1000, 2000, 3000, 4000, 5000]


# ============================================================
# 1. Identify algorithm and shift
# ============================================================

def identify_run(tb_dir):

    path = str(tb_dir).replace("\\", "/")

    if "/BC_SAC/" in path:
        algorithm = "BC_SAC"

    elif "/RLPD/" in path:
        algorithm = "RLPD"

    elif "/SAC_TARGET_ONLY/" in path:
        algorithm = "SAC_TARGET_ONLY"

    else:
        return None


    # Exclude old incorrect shift=2 directories
    if (
        "walker2d-friction-2/" in path
        or
        "walker2d-friction-srcdatatype-medium-2/" in path
    ):
        return None


    if "medium-0.5/" in path or "friction-0.5/" in path:
        shift = 0.5

    elif "medium-2.0/" in path or "friction-2.0/" in path:
        shift = 2.0

    else:
        return None


    run_name = tb_dir.parent.name

    if not run_name.startswith("r"):
        return None

    seed = int(run_name[1:])

    return algorithm, shift, seed


# ============================================================
# 2. Read the latest COMPLETE event file
# ============================================================

def read_complete_run(tb_dir):

    event_files = sorted(
        tb_dir.glob("events.out.tfevents*"),
        key=lambda x: x.stat().st_mtime
    )

    candidates = []

    for event_file in event_files:

        try:

            ea = EventAccumulator(
                str(event_file),
                size_guidance={"scalars": 0}
            )

            ea.Reload()

            if TAG not in ea.Tags().get("scalars", []):
                continue

            events = ea.Scalars(TAG)

            step_values = {
                e.step: e.value
                for e in events
                if e.step in TARGET_STEPS
            }

            if all(step in step_values for step in TARGET_STEPS):

                candidates.append(
                    (event_file, step_values)
                )

        except Exception:
            pass


    if not candidates:
        return None


    # Most recent complete run
    event_file, values = max(
        candidates,
        key=lambda x: x[0].stat().st_mtime
    )

    return event_file, values


# ============================================================
# 3. Collect the 18 valid runs
# ============================================================

rows = []


for tb_dir in ROOT.rglob("tb"):

    path = str(tb_dir).lower()

    if "walker2d" not in path:
        continue


    run_info = identify_run(tb_dir)

    if run_info is None:
        continue


    algorithm, shift, seed = run_info

    result = read_complete_run(tb_dir)

    if result is None:
        continue


    event_file, values = result


    for step in TARGET_STEPS:

        rows.append({
            "task": "walker2d",
            "algorithm": algorithm,
            "shift": shift,
            "seed": seed,
            "step": step,
            "normalized_score": values[step],
            "event_file": event_file.name
        })


df = pd.DataFrame(rows)


# ============================================================
# 4. Verify experiment structure
# ============================================================

print("\n============================================================")
print("walker2d DATA CHECK")
print("============================================================")

run_counts = (
    df.groupby(["algorithm", "shift"])["seed"]
      .nunique()
)

print(run_counts)

total_runs = (
    df[["algorithm", "shift", "seed"]]
    .drop_duplicates()
    .shape[0]
)

print("\nTotal valid runs:", total_runs)

if total_runs != 18:
    raise RuntimeError(
        f"Expected 18 valid runs, found {total_runs}"
    )


# ============================================================
# 5. Save all learning-curve observations
# ============================================================

df.to_csv(
    "rq2_walker2d_all_results.csv",
    index=False
)


# ============================================================
# 6. AUC for every individual seed
# ============================================================

auc_rows = []


for (algorithm, shift, seed), group in df.groupby(
    ["algorithm", "shift", "seed"]
):

    group = group.sort_values("step")

    x = group["step"].to_numpy()
    y = group["normalized_score"].to_numpy()

    auc = np.trapz(y, x)

    average_performance = auc / (
        TARGET_STEPS[-1] - TARGET_STEPS[0]
    )

    final_score = y[-1]


    auc_rows.append({
        "task": "walker2d",
        "algorithm": algorithm,
        "shift": shift,
        "seed": seed,
        "auc": auc,
        "average_performance": average_performance,
        "final_score": final_score
    })


auc_df = pd.DataFrame(auc_rows)

auc_df.to_csv(
    "rq2_walker2d_auc_by_seed.csv",
    index=False
)


# ============================================================
# 7. Mean ± SD across the three seeds
# ============================================================

summary = (
    auc_df
    .groupby(["task", "algorithm", "shift"])
    .agg(
        n=("seed", "count"),

        auc_mean=("auc", "mean"),
        auc_sd=("auc", "std"),

        average_performance_mean=(
            "average_performance", "mean"
        ),
        average_performance_sd=(
            "average_performance", "std"
        ),

        final_score_mean=("final_score", "mean"),
        final_score_sd=("final_score", "std")
    )
    .reset_index()
)


summary.to_csv(
    "rq2_walker2d_summary.csv",
    index=False
)


# ============================================================
# 8. Transfer gain relative to SAC_TARGET_ONLY
# ============================================================

transfer_rows = []


for shift in sorted(auc_df["shift"].unique()):

    shift_summary = summary[
        summary["shift"] == shift
    ]


    baseline_row = shift_summary[
        shift_summary["algorithm"] == "SAC_TARGET_ONLY"
    ]

    baseline = baseline_row[
        "average_performance_mean"
    ].iloc[0]


    for algorithm in ["BC_SAC", "RLPD"]:

        method_row = shift_summary[
            shift_summary["algorithm"] == algorithm
        ]

        performance = method_row[
            "average_performance_mean"
        ].iloc[0]


        transfer_gain = (
            (performance - baseline)
            / abs(baseline)
        ) * 100


        transfer_rows.append({
            "task": "walker2d",
            "shift": shift,
            "algorithm": algorithm,
            "method_performance": performance,
            "target_only_performance": baseline,
            "transfer_gain_percent": transfer_gain
        })


transfer_df = pd.DataFrame(transfer_rows)

transfer_df.to_csv(
    "rq2_walker2d_transfer_vs_target_only.csv",
    index=False
)


# ============================================================
# 9. Learning curves
# ============================================================

curve_summary = (
    df
    .groupby(["algorithm", "shift", "step"])
    ["normalized_score"]
    .agg(["mean", "std"])
    .reset_index()
)


for shift in sorted(df["shift"].unique()):

    plt.figure(figsize=(8, 5))

    temp = curve_summary[
        curve_summary["shift"] == shift
    ]


    for algorithm in [
        "SAC_TARGET_ONLY",
        "BC_SAC",
        "RLPD"
    ]:

        g = temp[
            temp["algorithm"] == algorithm
        ].sort_values("step")


        plt.plot(
            g["step"],
            g["mean"],
            marker="o",
            label=algorithm
        )


        plt.fill_between(
            g["step"],
            g["mean"] - g["std"],
            g["mean"] + g["std"],
            alpha=0.2
        )


    plt.xlabel("Target Environment Steps")
    plt.ylabel("Target Normalized Score")

    plt.title(
        f"walker2d Friction Shift = {shift}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"rq2_walker2d_learning_curve_shift_{shift}.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 10. Print results
# ============================================================

print("\n============================================================")
print("walker2d AUC SUMMARY")
print("============================================================")

print(
    summary[
        [
            "algorithm",
            "shift",
            "n",
            "auc_mean",
            "auc_sd",
            "average_performance_mean",
            "average_performance_sd",
            "final_score_mean",
            "final_score_sd"
        ]
    ].to_string(index=False)
)


print("\n============================================================")
print("TRANSFER VS SAC_TARGET_ONLY")
print("============================================================")

print(
    transfer_df.to_string(index=False)
)


print("\n============================================================")
print("FILES CREATED")
print("============================================================")

print("rq2_walker2d_all_results.csv")
print("rq2_walker2d_auc_by_seed.csv")
print("rq2_walker2d_summary.csv")
print("rq2_walker2d_transfer_vs_target_only.csv")
print("rq2_walker2d_learning_curve_shift_0.5.png")
print("rq2_walker2d_learning_curve_shift_2.0.png")