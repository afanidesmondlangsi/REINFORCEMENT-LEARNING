import pandas as pd
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = Path("matched_final_5000")
OUT = BASE / "ea_ntc_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE / "matched_final_288_long.csv")

ea = df[df["method"].isin([
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED"
])].copy()

diagnostic_tags = [
    "ea_ntc/adaptive_gate_strength",
    "ea_ntc/dynamics_updates",
    "ea_ntc/gate_ready",
    "ea_ntc/heldout_relative_mse",
    "ea_ntc/heldout_target_mse",
    "ea_ntc/heldout_zero_delta_mse",
    "ea_ntc/mean_source_weight",
    "ea_ntc/raw_mean_source_weight",
    "ea_ntc/source_weight_below_0_5",
    "ea_ntc/source_weight_ess",
    "ea_ntc/mean_uncertainty",
    "ea_ntc/mean_uncertainty_penalty",
    "ea_ntc/target_calibration_count",
    "ea_ntc/target_dynamics_train_count",
    "ea_ntc/target_uncertainty_reference",
    "test/target_return",
    "test/target_normalized_score"
]

checkpoints = [1000, 2000, 3000, 4000, 5000]

timeseries = []
checkpoint_rows = []
errors = []

print("\n========== EA-NTC DIAGNOSTIC EXTRACTION ==========")
print("EA runs:", len(ea))

for _, row in ea.iterrows():

    logdir = Path(row["path"])

    if not logdir.exists():
        errors.append({
            "method": row["method"],
            "environment": row["environment"],
            "shift": row["shift"],
            "seed": row["seed"],
            "error": "TensorBoard directory missing",
            "path": str(logdir)
        })
        continue

    try:
        acc = EventAccumulator(
            str(logdir),
            size_guidance={"scalars": 0}
        )
        acc.Reload()

        available = acc.Tags().get("scalars", [])

        for tag in diagnostic_tags:

            if tag not in available:
                continue

            events = acc.Scalars(tag)

            for event in events:
                timeseries.append({
                    "method": row["method"],
                    "environment": row["environment"],
                    "shift": row["shift"],
                    "seed": row["seed"],
                    "tag": tag,
                    "step": event.step,
                    "value": event.value
                })

            for cp in checkpoints:
                eligible = [e for e in events if e.step <= cp]

                if not eligible:
                    continue

                event = max(eligible, key=lambda e: e.step)

                checkpoint_rows.append({
                    "method": row["method"],
                    "environment": row["environment"],
                    "shift": row["shift"],
                    "seed": row["seed"],
                    "checkpoint": cp,
                    "tag": tag,
                    "observed_step": event.step,
                    "value": event.value
                })

    except Exception as exc:
        errors.append({
            "method": row["method"],
            "environment": row["environment"],
            "shift": row["shift"],
            "seed": row["seed"],
            "error": str(exc),
            "path": str(logdir)
        })

ts = pd.DataFrame(timeseries)
cp_df = pd.DataFrame(checkpoint_rows)
err_df = pd.DataFrame(errors)

if not ts.empty:
    ts.to_csv(OUT / "ea_ntc_diagnostics_timeseries.csv", index=False)

if not cp_df.empty:

    cp_df.to_csv(
        OUT / "ea_ntc_diagnostics_checkpoints.csv",
        index=False
    )

    summary = (
        cp_df.groupby([
            "environment", "shift", "method",
            "checkpoint", "tag"
        ])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.to_csv(
        OUT / "ea_ntc_diagnostics_summary.csv",
        index=False
    )

if not err_df.empty:
    err_df.to_csv(OUT / "ea_ntc_extraction_errors.csv", index=False)

print("\n========== EXTRACTION SUMMARY ==========")
print("Runs:", len(ea))
print("Successful time-series records:", len(ts))
print("Checkpoint records:", len(cp_df))
print("Errors:", len(err_df))

print("\nAvailable diagnostic tags:")
if not ts.empty:
    print(ts["tag"].value_counts().to_string())

print("\nOutput directory:", OUT.resolve())
