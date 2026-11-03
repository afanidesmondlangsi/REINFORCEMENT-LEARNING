from pathlib import Path
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OLD = Path("logs_ea_ntc_norm_vs_mean_4env_6shift_6seed_5000")
SOURCES = {
    "hopper": Path("logs_hopper_corrected_gate_48runs"),
    "walker2d": Path("logs_walker2d_revised_gate_48runs"),
    "halfcheetah": OLD,
    "ant": OLD,
}

SHIFTS = ["0.1", "0.5", "2.0", "5.0"]
SEEDS = range(6)
VARIANTS = ["NORMALIZED", "UNNORMALIZED"]
OUT = Path("analysis_correct_runs_4env")
OUT.mkdir(exist_ok=True)

audit = []
curves = []

for env, root in SOURCES.items():
    for shift in SHIFTS:
        for seed in SEEDS:
            for variant in VARIANTS:
                name = f"{variant}_{env}_shift_{shift}_seed_{seed}"
                folder = root / name

                record = dict(
                    environment=env,
                    cohort="revised" if env in ("hopper", "walker2d") else "original",
                    variant=variant,
                    shift=shift,
                    seed=seed,
                    directory=str(folder),
                    valid=False,
                    success_marker=(folder / "_SUCCESS.txt").exists()
                )

                files = list(folder.rglob("events.out.tfevents.*"))
                directories = set(p.parent for p in files)

                if len(directories) != 1:
                    record["error"] = f"Event directories: {len(directories)}"
                    audit.append(record)
                    continue

                try:
                    ea = EventAccumulator(
                        str(next(iter(directories))),
                        size_guidance={"scalars": 0}
                    )
                    ea.Reload()
                    tags = set(ea.Tags().get("scalars", []))

                    score_tag = "test/target_normalized_score"
                    if score_tag not in tags:
                        raise ValueError("Missing test score")

                    score_map = {
                        int(x.step): float(x.value)
                        for x in ea.Scalars(score_tag)
                    }

                    required = [1000, 2000, 3000, 4000, 5000]
                    if not all(x in score_map for x in required):
                        raise ValueError("Missing evaluation checkpoints")

                    scores = np.array(
                        [score_map[x] for x in required]
                    )

                    if not np.isfinite(scores).all():
                        raise ValueError("Non-finite scores")

                    record["final_score"] = score_map[5000]
                    record["auc"] = float(
                        np.trapz(scores, np.array(required))
                    )
                    record["early_1000"] = score_map[1000]
                    record["early_2000"] = score_map[2000]
                    record["valid"] = True

                    diagnostics = {
                        "gate_ready": "ea_ntc/gate_ready",
                        "gate_strength": "ea_ntc/adaptive_gate_strength",
                        "source_weight": "ea_ntc/mean_source_weight",
                        "relative_mse": "ea_ntc/heldout_relative_mse"
                    }

                    for column, tag in diagnostics.items():
                        if tag in tags:
                            values = ea.Scalars(tag)
                            if values:
                                record[column] = float(values[-1].value)

                    for step in required:
                        curves.append(dict(
                            environment=env,
                            cohort=record["cohort"],
                            variant=variant,
                            shift=shift,
                            seed=seed,
                            step=step,
                            score=score_map[step]
                        ))

                except Exception as exc:
                    record["error"] = str(exc)

                audit.append(record)

audit = pd.DataFrame(audit)
curves = pd.DataFrame(curves)

audit.to_csv(OUT / "run_audit.csv", index=False)
curves.to_csv(OUT / "learning_curves.csv", index=False)

print("\nRUN VALIDATION")
print(
    audit.groupby(["environment", "variant"])["valid"]
    .agg(["sum", "count"]).to_string()
)

if not audit["valid"].all():
    print("\nINVALID RUNS")
    print(
        audit.loc[~audit.valid,
                  ["environment", "variant", "shift", "seed", "error"]]
        .to_string(index=False)
    )
    raise SystemExit("Audit incomplete. See run_audit.csv.")

normalized = audit[audit.variant == "NORMALIZED"]
unnormalized = audit[audit.variant == "UNNORMALIZED"]

paired = normalized.merge(
    unnormalized,
    on=["environment", "shift", "seed"],
    suffixes=("_norm", "_mean"),
    validate="one_to_one"
)

for metric in ["auc", "final_score", "early_1000", "early_2000"]:
    paired["delta_" + metric] = (
        paired[metric + "_norm"] - paired[metric + "_mean"]
    )

paired.to_csv(OUT / "paired_results.csv", index=False)

summary = (
    paired.groupby(["environment", "shift"])
    .agg(
        n=("seed", "count"),
        mean_delta_auc=("delta_auc", "mean"),
        sd_delta_auc=("delta_auc", "std"),
        mean_delta_final=("delta_final_score", "mean"),
        mean_delta_1000=("delta_early_1000", "mean"),
        mean_delta_2000=("delta_early_2000", "mean"),
        auc_wins=("delta_auc", lambda x: int((x > 0).sum())),
        final_wins=("delta_final_score", lambda x: int((x > 0).sum()))
    )
    .reset_index()
)

summary.to_csv(OUT / "summary_by_environment_shift.csv", index=False)

print("\nPAIRED RESULTS")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nGATE DIAGNOSTICS")
for env, subset in audit.groupby("environment"):
    print(f"\n{env}")
    for column in ["gate_ready", "gate_strength", "source_weight", "relative_mse"]:
        if column in subset.columns:
            values = subset[column].dropna()
            if len(values):
                print(
                    column,
                    "mean=", round(values.mean(), 5),
                    "min=", round(values.min(), 5),
                    "max=", round(values.max(), 5)
                )

print("\nResults saved to:", OUT)
