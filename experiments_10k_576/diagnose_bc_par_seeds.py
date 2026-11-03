import csv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("experiments_10k_576/results/BC_PAR/bc_par")
OUT = Path("experiments_10k_576/BC_PAR_diagnostics")

OUT.mkdir(parents=True, exist_ok=True)

ENVIRONMENTS = [
    "hopper-friction",
    "halfcheetah-friction",
    "walker2d-friction",
    "ant-friction"
]

SHIFTS = ["0.1", "0.5", "2.0", "5.0"]
SEEDS = range(6)

STEPS = np.arange(1000, 10001, 1000)

TAGS = [
    "test/target normalized score",
    "test/target_normalized_score"
]

def calculate_auc(scores):
    return float(np.trapz(scores, STEPS) / 9000.0)

def robust_outlier_flags(values):
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad > 0:
        robust_z = 0.6745 * (values - median) / mad
    else:
        robust_z = np.zeros(len(values))

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    if iqr > 0:
        iqr_flag = (values < q1 - 1.5 * iqr) | (
            values > q3 + 1.5 * iqr
        )
    else:
        iqr_flag = np.zeros(len(values), dtype=bool)

    return robust_z, iqr_flag

all_seed_results = []
curve_groups = defaultdict(list)
errors = []

# ============================================================
# LOAD ALL 96 ORIGINAL TENSORBOARD LEARNING CURVES
# ============================================================

for env in ENVIRONMENTS:
    for shift in SHIFTS:

        folder = f"{env}-srcdatatype-medium-{shift}"

        for seed in SEEDS:

            run_dir = ROOT / folder / f"r{seed}"
            tb_dir = run_dir / "tb"

            try:
                ea = EventAccumulator(
                    str(tb_dir),
                    size_guidance={"scalars": 0}
                )
                ea.Reload()

                tags = ea.Tags().get("scalars", [])
                tag = next((t for t in TAGS if t in tags), None)

                if tag is None:
                    raise ValueError("Normalized score tag missing")

                events = ea.Scalars(tag)

                values = {}

                for event in events:
                    if event.step in STEPS:
                        values[event.step] = event.value

                if set(values) != set(STEPS):
                    raise ValueError("Missing evaluation steps")

                scores = np.array(
                    [values[int(step)] for step in STEPS],
                    dtype=float
                )

                if not np.all(np.isfinite(scores)):
                    raise ValueError("Non-finite scores")

                auc = calculate_auc(scores)

                result = {
                    "environment": env,
                    "shift": shift,
                    "seed": seed,
                    "normalized_auc": auc,
                    "final_score": float(scores[-1]),
                    "early_mean": float(scores[:5].mean()),
                    "late_mean": float(scores[5:].mean()),
                    "late_minus_early": float(
                        scores[5:].mean() - scores[:5].mean()
                    )
                }

                all_seed_results.append(result)

                curve_groups[(env, shift)].append({
                    "seed": seed,
                    "scores": scores,
                    "auc": auc
                })

            except Exception as exc:
                errors.append({
                    "environment": env,
                    "shift": shift,
                    "seed": seed,
                    "error": str(exc)
                })

if len(all_seed_results) != 96:
    raise RuntimeError(
        f"Expected 96 valid runs, found {len(all_seed_results)}. "
        "Resolve missing data before interpreting results."
    )

# ============================================================
# SEED-LEVEL AUC AND FINAL-SCORE OUTLIER ANALYSIS
# ============================================================

seed_diagnostics = []

for key, results in sorted(curve_groups.items()):

    env, shift = key

    for metric in ["normalized_auc", "final_score"]:

        vals = [r[metric] for r in all_seed_results
                if r["environment"] == env
                and r["shift"] == shift]

        robust_z, iqr_flags = robust_outlier_flags(vals)

        for i, seed in enumerate(SEEDS):

            matching = next(
                r for r in all_seed_results
                if r["environment"] == env
                and r["shift"] == shift
                and r["seed"] == seed
            )

            seed_diagnostics.append({
                "environment": env,
                "shift": shift,
                "seed": seed,
                "metric": metric,
                "value": matching[metric],
                "robust_z": float(robust_z[i]),
                "iqr_outlier": bool(iqr_flags[i]),
                "robust_z_outlier": bool(abs(robust_z[i]) > 3.5)
            })

# ============================================================
# GROUP SUMMARY: MEAN, MEDIAN, SD, IQR
# ============================================================

group_summary = []

for (env, shift), results in sorted(curve_groups.items()):

    aucs = np.array([r["auc"] for r in results])
    finals = np.array([r["scores"][-1] for r in results])

    group_summary.append({
        "environment": env,
        "shift": shift,
        "n_seeds": len(results),
        "auc_mean": float(aucs.mean()),
        "auc_sd": float(aucs.std(ddof=1)),
        "auc_median": float(np.median(aucs)),
        "auc_q1": float(np.percentile(aucs, 25)),
        "auc_q3": float(np.percentile(aucs, 75)),
        "final_mean": float(finals.mean()),
        "final_sd": float(finals.std(ddof=1)),
        "final_median": float(np.median(finals)),
        "final_q1": float(np.percentile(finals, 25)),
        "final_q3": float(np.percentile(finals, 75)),
        "early_mean": float(np.mean([
            r["scores"][:5].mean() for r in results
        ])),
        "late_mean": float(np.mean([
            r["scores"][5:].mean() for r in results
        ])),
        "late_minus_early": float(np.mean([
            (r["scores"][5:].mean() - r["scores"][:5].mean()) for r in results
        ]))
    })

# ============================================================
# LEARNING CURVE SUMMARY AT EACH EVALUATION STEP
# ============================================================

curve_summary = []

for (env, shift), results in sorted(curve_groups.items()):

    matrix = np.vstack([r["scores"] for r in results])

    for i, step in enumerate(STEPS):

        curve_summary.append({
            "environment": env,
            "shift": shift,
            "step": int(step),
            "mean_score": float(matrix[:, i].mean()),
            "median_score": float(np.median(matrix[:, i])),
            "sd_score": float(matrix[:, i].std(ddof=1)),
            "min_score": float(matrix[:, i].min()),
            "max_score": float(matrix[:, i].max())
        })

def save_csv(filename, rows):
    path = OUT / filename

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", path)

save_csv("seed_diagnostics.csv", seed_diagnostics)
save_csv("group_summary.csv", group_summary)
save_csv("learning_curve_summary.csv", curve_summary)
save_csv("individual_seed_results.csv", all_seed_results)

# ============================================================
# PLOT 1: SIX-SEED LEARNING CURVES, FOUR SHIFTS PER ENVIRONMENT
# ============================================================

for env in ENVIRONMENTS:

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for j, shift in enumerate(SHIFTS):

        ax = axes[j]
        results = curve_groups[(env, shift)]

        matrix = np.vstack([r["scores"] for r in results])

        for r in results:
            ax.plot(
                STEPS,
                r["scores"],
                alpha=0.40,
                linewidth=1.1,
                marker="."
            )

        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)

        ax.plot(
            STEPS,
            mean,
            linewidth=2.5,
            marker="o",
            label="Mean across 6 seeds"
        )

        ax.fill_between(
            STEPS,
            mean - sd,
            mean + sd,
            alpha=0.15,
            label="Mean +/- 1 SD"
        )

        ax.set_title(f"Friction shift {shift}")
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Target normalized score")
        ax.set_xticks(STEPS)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle(
        f"BC-PAR: Six-seed learning curves - {env}",
        fontsize=15
    )

    fig.tight_layout()

    filename = OUT / f"{env}_learning_curves.png"
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", filename)

# ============================================================
# PLOT 2: AUC DISTRIBUTIONS BY ENVIRONMENT AND SHIFT
# ============================================================

for env in ENVIRONMENTS:

    data = [
        [
            r["auc"]
            for r in curve_groups[(env, shift)]
        ]
        for shift in SHIFTS
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.boxplot(data, labels=SHIFTS, showmeans=True)

    ax.set_title(f"BC-PAR AUC distribution - {env}")
    ax.set_xlabel("Friction shift")
    ax.set_ylabel("Normalized AUC")
    ax.grid(True, axis="y", alpha=0.25)

    filename = OUT / f"{env}_auc_distribution.png"
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", filename)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 90)
print("                 BC-PAR SEED AND LEARNING-CURVE DIAGNOSTICS")
print("=" * 90)

print("Valid runs:", len(all_seed_results))
print("Invalid runs:", len(errors))

print("\nGROUP-LEVEL AUC: MEAN VS MEDIAN")
print("-" * 90)

for r in group_summary:
    print(
        f"{r['environment']:22s} shift={r['shift']:4s} "
        f"Mean={r['auc_mean']:.4f} "
        f"Median={r['auc_median']:.4f} "
        f"SD={r['auc_sd']:.4f} "
        f"Early={r['early_mean']:.4f} "
        f"Late={r['late_mean']:.4f}"
    )

print("\nOUTLIER FLAGS")
print("-" * 90)

flagged = [
    r for r in seed_diagnostics
    if r["iqr_outlier"] or r["robust_z_outlier"]
]

if not flagged:
    print("No outlier flags detected.")
else:
    for r in flagged:
        print(
            f"{r['environment']:22s} shift={r['shift']:4s} "
            f"seed={r['seed']} metric={r['metric']} "
            f"value={r['value']:.4f} "
            f"robust_z={r['robust_z']:.2f} "
            f"IQR={r['iqr_outlier']}"
        )

print("\nDIAGNOSTIC FILES:")
print(OUT.resolve())

print("\nAnalysis completed.")
print("Outlier flags are diagnostic indicators, not automatic exclusions.")


