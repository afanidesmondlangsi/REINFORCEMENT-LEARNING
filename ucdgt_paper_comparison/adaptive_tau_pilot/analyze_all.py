from pathlib import Path
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path("ucdgt_paper_comparison")

RUNS = ROOT / "adaptive_tau_pilot" / "runs"

FIXED_CSV = (
    ROOT / "gate_temperature_ablation"
    / "analysis"
    / "temperature_run_metrics.csv"
)

OUT = ROOT / "adaptive_tau_pilot" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

ENVIRONMENTS = [
    "ant-friction",
    "halfcheetah-friction",
    "hopper-friction",
    "walker2d-friction",
]

SHIFTS = [0.5, 2.0]
SEEDS = [0, 1, 2]
TEMPERATURES = [5.0, 10.0, 20.0]

SCORE_TAG = "test/target_normalized_score"

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def get_last_scalar(ea, tag):
    tags = ea.Tags().get("scalars", [])

    if tag not in tags:
        return float("nan")

    values = ea.Scalars(tag)

    if not values:
        return float("nan")

    return float(values[-1].value)


def get_scalar_series(ea, tag):
    tags = ea.Tags().get("scalars", [])

    if tag not in tags:
        return []

    return ea.Scalars(tag)


# ============================================================
# 3. EXTRACT ALL ADAPTIVE RUNS
# ============================================================

rows = []
learning_curves = []
controller_trajectories = []

expected_steps = set(range(1000, 10001, 1000))

for env in ENVIRONMENTS:
    for shift in SHIFTS:
        for seed in SEEDS:

            run_dir = (
                RUNS
                / "ucdgt_adaptive_tau"
                / f"{env}-srcdatatype-medium-{shift}"
                / f"r{seed}"
            )

            event_files = list(
                run_dir.rglob("events.out.tfevents.*")
            )

            if len(event_files) != 1:
                raise RuntimeError(
                    f"Expected exactly one event file for "
                    f"{env}, shift={shift}, seed={seed}. "
                    f"Found {len(event_files)}."
                )

            ea = EventAccumulator(
                str(event_files[0]),
                size_guidance={"scalars": 0}
            )
            ea.Reload()

            tags = ea.Tags().get("scalars", [])

            if SCORE_TAG not in tags:
                raise RuntimeError(
                    f"Missing score tag in {run_dir}. "
                    f"Available tags: {tags}"
                )

            scores_by_step = {
                int(v.step): float(v.value)
                for v in ea.Scalars(SCORE_TAG)
                if 1000 <= v.step <= 10000
            }

            if set(scores_by_step) != expected_steps:
                raise RuntimeError(
                    f"Incomplete evaluations for "
                    f"{env}, shift={shift}, seed={seed}. "
                    f"Found: {sorted(scores_by_step)}"
                )

            steps = np.array(
                sorted(scores_by_step),
                dtype=float
            )

            scores = np.array(
                [scores_by_step[int(s)] for s in steps],
                dtype=float
            )

            # Match the earlier fixed-temperature AUC convention.
            auc = float(np.trapz(scores, steps))

            final_score = float(scores[-1])
            peak_score = float(scores.max())

            final_tau = get_last_scalar(
                ea,
                "ucdgt_adaptive_tau/temperature"
            )

            final_ess = get_last_scalar(
                ea,
                "ucdgt_adaptive_tau/source_ess_fraction"
            )

            update_count = get_last_scalar(
                ea,
                "ucdgt_adaptive_tau/update_count"
            )

            rows.append({
                "environment": env,
                "shift": shift,
                "seed": seed,
                "auc": auc,
                "final_score": final_score,
                "peak_score": peak_score,
                "final_tau": final_tau,
                "final_ess": final_ess,
                "temperature_updates": update_count,
            })

            # Save individual learning curves.
            for step, score in zip(steps, scores):
                learning_curves.append({
                    "environment": env,
                    "shift": shift,
                    "seed": seed,
                    "step": int(step),
                    "normalized_score": float(score),
                })

            # Save temperature and ESS trajectories.
            trajectory_tags = {
                "temperature":
                    "ucdgt_adaptive_tau/temperature",
                "ess_fraction":
                    "ucdgt_adaptive_tau/source_ess_fraction",
                "update_count":
                    "ucdgt_adaptive_tau/update_count",
            }

            for metric, tag in trajectory_tags.items():
                for event in get_scalar_series(ea, tag):
                    controller_trajectories.append({
                        "environment": env,
                        "shift": shift,
                        "seed": seed,
                        "metric": metric,
                        "training_iteration": int(event.step),
                        "value": float(event.value),
                    })

# ============================================================
# 4. SAVE ADAPTIVE RESULTS
# ============================================================

adaptive = pd.DataFrame(rows)

adaptive.to_csv(
    OUT / "all_adaptive_run_metrics.csv",
    index=False
)

pd.DataFrame(learning_curves).to_csv(
    OUT / "all_adaptive_learning_curves.csv",
    index=False
)

pd.DataFrame(controller_trajectories).to_csv(
    OUT / "all_adaptive_controller_trajectories.csv",
    index=False
)

print("\n" + "=" * 90)
print("ADAPTIVE UCDGT: ALL 24 RUNS")
print("=" * 90)

print(adaptive.to_string(index=False))

print("\nTotal adaptive runs:", len(adaptive))

# ============================================================
# 5. SUMMARY BY ENVIRONMENT AND SHIFT
# ============================================================

summary = adaptive.groupby(
    ["environment", "shift"]
).agg(
    mean_auc=("auc", "mean"),
    sd_auc=("auc", "std"),
    mean_final_score=("final_score", "mean"),
    mean_peak_score=("peak_score", "mean"),
    mean_final_tau=("final_tau", "mean"),
    mean_final_ess=("final_ess", "mean"),
    mean_updates=("temperature_updates", "mean"),
).reset_index()

summary["auc_cv_pct"] = np.where(
    summary["mean_auc"].abs() > 1e-12,
    100 * summary["sd_auc"]
    / summary["mean_auc"].abs(),
    np.nan
)

summary.to_csv(
    OUT / "all_adaptive_summary.csv",
    index=False
)

print("\n" + "=" * 90)
print("ADAPTIVE SUMMARY BY ENVIRONMENT AND SHIFT")
print("=" * 90)

print(summary.to_string(index=False))

# ============================================================
# 6. LOAD AVAILABLE FIXED-TEMPERATURE BASELINES
# ============================================================

if FIXED_CSV.exists():

    fixed = pd.read_csv(FIXED_CSV)

    required = {
        "environment",
        "shift",
        "seed",
        "tau",
        "auc",
    }

    missing_columns = required - set(fixed.columns)

    if missing_columns:
        raise RuntimeError(
            f"Fixed baseline CSV missing columns: "
            f"{sorted(missing_columns)}"
        )

    for col in ["shift", "seed", "tau", "auc"]:
        fixed[col] = pd.to_numeric(
            fixed[col],
            errors="raise"
        )

else:

    print(
        "\nWARNING: Fixed-temperature baseline CSV "
        "does not exist. Adaptive diagnostics will "
        "still be saved."
    )

    fixed = pd.DataFrame(
        columns=[
            "environment",
            "shift",
            "seed",
            "tau",
            "auc",
        ]
    )

# ============================================================
# 7. MATCHED ADAPTIVE VS FIXED COMPARISONS
# ============================================================

comparisons = []
paired_rows = []
missing_baselines = []

for env in ENVIRONMENTS:
    for shift in SHIFTS:

        ad = adaptive[
            (adaptive["environment"] == env) &
            np.isclose(adaptive["shift"], shift)
        ][["seed", "auc"]].rename(
            columns={"auc": "adaptive_auc"}
        )

        for tau in TEMPERATURES:

            fx = fixed[
                (fixed["environment"] == env) &
                np.isclose(fixed["shift"], shift) &
                np.isclose(fixed["tau"], tau)
            ][["seed", "auc"]].rename(
                columns={"auc": "fixed_auc"}
            )

            expected_seeds = set(SEEDS)

            found_seeds = set(
                pd.to_numeric(
                    fx["seed"],
                    errors="coerce"
                ).dropna().astype(int)
            )

            if (
                len(fx) != len(SEEDS)
                or found_seeds != expected_seeds
            ):

                missing_baselines.append({
                    "environment": env,
                    "shift": shift,
                    "fixed_tau": tau,
                    "baseline_runs_found": len(fx),
                    "seeds_found": ",".join(
                        str(s) for s in sorted(found_seeds)
                    ),
                })

                continue

            paired = ad.merge(
                fx,
                on="seed",
                validate="one_to_one"
            ).sort_values("seed")

            if len(paired) != len(SEEDS):
                raise RuntimeError(
                    f"Seed mismatch for {env}, "
                    f"shift={shift}, tau={tau}"
                )

            paired["delta_auc"] = (
                paired["adaptive_auc"]
                - paired["fixed_auc"]
            )

            adaptive_mean = paired["adaptive_auc"].mean()
            fixed_mean = paired["fixed_auc"].mean()
            mean_delta = paired["delta_auc"].mean()

            relative_pct = (
                100 * mean_delta / fixed_mean
                if abs(fixed_mean) > 1e-12
                else np.nan
            )

            comparisons.append({
                "environment": env,
                "shift": shift,
                "fixed_tau": tau,
                "adaptive_mean_auc": adaptive_mean,
                "fixed_mean_auc": fixed_mean,
                "mean_delta_auc": mean_delta,
                "relative_improvement_pct": relative_pct,
                "adaptive_wins": int(
                    (paired["delta_auc"] > 0).sum()
                ),
                "total_seeds": len(paired),
            })

            for _, row in paired.iterrows():
                paired_rows.append({
                    "environment": env,
                    "shift": shift,
                    "fixed_tau": tau,
                    "seed": int(row["seed"]),
                    "adaptive_auc": float(row["adaptive_auc"]),
                    "fixed_auc": float(row["fixed_auc"]),
                    "delta_auc": float(row["delta_auc"]),
                })

comparison_columns = [
    "environment", "shift", "fixed_tau",
    "adaptive_mean_auc", "fixed_mean_auc",
    "mean_delta_auc", "relative_improvement_pct",
    "adaptive_wins", "total_seeds"
]

comparison = pd.DataFrame(
    comparisons,
    columns=comparison_columns
)

missing_df = pd.DataFrame(
    missing_baselines,
    columns=[
        "environment",
        "shift",
        "fixed_tau",
        "baseline_runs_found",
        "seeds_found",
    ]
)

paired_df = pd.DataFrame(
    paired_rows,
    columns=[
        "environment", "shift", "fixed_tau",
        "seed", "adaptive_auc", "fixed_auc",
        "delta_auc"
    ]
)

comparison.to_csv(
    OUT / "all_adaptive_vs_fixed.csv",
    index=False
)

paired_df.to_csv(
    OUT / "all_adaptive_paired_seed_differences.csv",
    index=False
)

missing_df.to_csv(
    OUT / "missing_fixed_temperature_baselines.csv",
    index=False
)

# ============================================================
# 8. DISPLAY COMPARISONS
# ============================================================

print("\n" + "=" * 90)
print("AVAILABLE MATCHED FIXED-TEMPERATURE COMPARISONS")
print("=" * 90)

if comparison.empty:
    print("No complete matched fixed-temperature comparisons.")
else:
    print(comparison.to_string(index=False))

print("\n" + "=" * 90)
print("MISSING FIXED-TEMPERATURE BASELINES")
print("=" * 90)

if missing_df.empty:
    print("All fixed-temperature baselines are available.")
else:
    print(missing_df.to_string(index=False))

# ============================================================
# 9. FINAL CONTROLLER DIAGNOSTICS
# ============================================================

print("\n" + "=" * 90)
print("CONTROLLER DIAGNOSTICS")
print("=" * 90)

diagnostics = summary[
    [
        "environment",
        "shift",
        "mean_final_tau",
        "mean_final_ess",
        "mean_updates",
        "auc_cv_pct",
    ]
].copy()

diagnostics["tau_change_from_initial"] = (
    diagnostics["mean_final_tau"] - 10.0
)

diagnostics["ess_deviation_from_target"] = (
    diagnostics["mean_final_ess"] - 0.85
)

diagnostics.to_csv(
    OUT / "all_controller_diagnostics.csv",
    index=False
)

print(diagnostics.to_string(index=False))

# ============================================================
# 10. COMPLETION SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("FINAL ANALYSIS COMPLETED")
print("=" * 90)

print("Adaptive runs analyzed:", len(adaptive))
print("Environment-shift conditions:", len(summary))
print("Available matched comparisons:", len(comparison))
print("Missing fixed-baseline conditions:", len(missing_df))

print("\nFiles saved in:", OUT.resolve())

for name in [
    "all_adaptive_run_metrics.csv",
    "all_adaptive_summary.csv",
    "all_adaptive_learning_curves.csv",
    "all_adaptive_controller_trajectories.csv",
    "all_adaptive_vs_fixed.csv",
    "all_adaptive_paired_seed_differences.csv",
    "missing_fixed_temperature_baselines.csv",
    "all_controller_diagnostics.csv",
]:
    print(" -", name)
