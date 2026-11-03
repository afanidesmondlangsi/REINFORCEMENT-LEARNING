import csv
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("experiments_10k_576/results/BC_PAR/bc_par")
OUT = Path("experiments_10k_576")

ENVIRONMENTS = [
    "hopper-friction",
    "halfcheetah-friction",
    "walker2d-friction",
    "ant-friction"
]
SHIFTS = ["0.1", "0.5", "2.0", "5.0"]
SEEDS = range(6)

EXPECTED_STEPS = np.arange(1000, 10001, 1000)

TAGS = [
    "test/target normalized score",
    "test/target_normalized_score"
]

individual = []
groups = defaultdict(list)
errors = []

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
                    raise ValueError("Target normalized score tag missing")

                events = ea.Scalars(tag)

                # Keep the latest recorded value for each evaluation step.
                values = {}
                for event in events:
                    if event.step in EXPECTED_STEPS:
                        values[event.step] = event.value

                missing = sorted(set(EXPECTED_STEPS) - set(values))

                if missing:
                    raise ValueError(f"Missing steps: {missing}")

                scores = np.array(
                    [values[int(step)] for step in EXPECTED_STEPS],
                    dtype=float
                )

                if not np.all(np.isfinite(scores)):
                    raise ValueError("Non-finite evaluation scores")

                raw_auc = float(np.trapz(scores, EXPECTED_STEPS))
                normalized_auc = raw_auc / (
                    EXPECTED_STEPS[-1] - EXPECTED_STEPS[0]
                )

                final_score = float(scores[-1])
                early_mean = float(scores[:5].mean())
                overall_mean = float(scores.mean())

                result = {
                    "environment": env,
                    "shift": shift,
                    "seed": seed,
                    "raw_auc": raw_auc,
                    "normalized_auc": normalized_auc,
                    "final_score": final_score,
                    "early_mean_score": early_mean,
                    "mean_score": overall_mean,
                    "min_score": float(scores.min()),
                    "max_score": float(scores.max()),
                    "status": "VALID"
                }

                individual.append(result)
                groups[(env, shift)].append(result)

            except Exception as exc:
                errors.append({
                    "environment": env,
                    "shift": shift,
                    "seed": seed,
                    "error": str(exc)
                })

# Save individual results.
individual_path = OUT / "BC_PAR_auc_individual.csv"

with individual_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=individual[0].keys())
    writer.writeheader()
    writer.writerows(individual)

# Aggregate six seeds for each environment and shift.
summary = []

metrics = [
    "raw_auc",
    "normalized_auc",
    "final_score",
    "early_mean_score",
    "mean_score"
]

for (env, shift), results in sorted(groups.items()):

    if len(results) != 6:
        continue

    row = {
        "environment": env,
        "shift": shift,
        "seeds": len(results)
    }

    for metric in metrics:
        vals = np.array([r[metric] for r in results])

        row[f"{metric}_mean"] = float(vals.mean())
        row[f"{metric}_std"] = float(vals.std(ddof=1))

    summary.append(row)

summary_path = OUT / "BC_PAR_auc_summary.csv"

if summary:
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

# Print results.
print("\n" + "=" * 85)
print("                 BC-PAR AUC ANALYSIS")
print("=" * 85)

print(f"Valid individual runs: {len(individual)}/96")
print(f"Invalid runs: {len(errors)}")
print(f"Complete environment-shift groups: {len(summary)}/16")

print("\nENVIRONMENT / SHIFT SUMMARY")
print("-" * 85)

for r in summary:
    print(
        f"{r['environment']:22s} "
        f"shift={r['shift']:4s} "
        f"AUC={r['normalized_auc_mean']:.4f} "
        f"+/- {r['normalized_auc_std']:.4f} "
        f"Final={r['final_score_mean']:.4f} "
        f"+/- {r['final_score_std']:.4f}"
    )

if errors:
    print("\nINVALID RUNS")
    for e in errors:
        print(e)

print("\nFILES SAVED:")
print(individual_path)
print(summary_path)

print("\nAUC definition: trapezoidal area over steps 1000-10000.")
print("Normalized AUC: raw AUC divided by 9000.")

if len(individual) == 96 and len(summary) == 16 and not errors:
    print("\nPASS: All 96 runs included in AUC analysis.")
else:
    print("\nWARNING: Investigate missing or invalid AUC results.")
