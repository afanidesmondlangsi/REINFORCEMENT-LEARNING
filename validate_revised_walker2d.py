import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("logs_walker2d_revised_gate_48runs")
OUT = Path("matched_final_5000/walker2d_revised_validation")
OUT.mkdir(parents=True, exist_ok=True)

SHIFTS = ["0.1", "0.5", "2.0", "5.0"]
SEEDS = [0, 1, 2]
CHECKPOINTS = [1000, 2000, 3000, 4000, 5000]

TREATMENTS = {
    "NORMALIZED": "normalized",
    "UNNORMALIZED": "mean"
}

TAG = "test/target_return"

records = []
errors = []

print("\n========== REVISED WALKER2D VALIDATION ==========")

for treatment, reduction in TREATMENTS.items():
    for shift in SHIFTS:
        for seed in SEEDS:

            name = f"{treatment}_walker2d_shift_{shift}_seed_{seed}"
            run_dir = ROOT / name

            if not run_dir.exists():
                errors.append((name, "Missing run directory"))
                continue

            event_files = list(run_dir.rglob("events.out.tfevents.*"))

            if not event_files:
                errors.append((name, "Missing TensorBoard events"))
                continue

            event_dirs = sorted(set(p.parent for p in event_files))

            all_events = []

            for event_dir in event_dirs:
                try:
                    acc = EventAccumulator(
                        str(event_dir),
                        size_guidance={"scalars": 0}
                    )
                    acc.Reload()

                    if TAG in acc.Tags().get("scalars", []):
                        all_events.extend(acc.Scalars(TAG))

                except Exception as exc:
                    errors.append((name, str(exc)))

            if not all_events:
                errors.append((name, "Missing target return tag"))
                continue

            # Keep the latest event for each evaluation step.
            by_step = {}
            for event in all_events:
                if event.step <= 5000:
                    if (event.step not in by_step or
                        event.wall_time > by_step[event.step].wall_time):
                        by_step[event.step] = event

            values = {}
            observed = {}

            for cp in CHECKPOINTS:
                eligible = [
                    (step, event)
                    for step, event in by_step.items()
                    if step <= cp
                ]

                if not eligible:
                    values[cp] = None
                    observed[cp] = None
                else:
                    step, event = max(eligible, key=lambda x: x[0])
                    values[cp] = event.value
                    observed[cp] = step

            if any(values[cp] is None for cp in CHECKPOINTS):
                errors.append((name, f"Incomplete checkpoints: {observed}"))
                continue

            records.append({
                "treatment": treatment,
                "reduction": reduction,
                "environment": "walker2d",
                "shift": float(shift),
                "seed": seed,
                "path": str(event_dirs[0]),
                **{f"eval_{cp}": values[cp] for cp in CHECKPOINTS}
            })

            print(
                f"VALIDATED: {name} | "
                f"1000={values[1000]:.3f} | "
                f"5000={values[5000]:.3f}"
            )

df = pd.DataFrame(records)

if not df.empty:
    df.to_csv(
        OUT / "walker2d_revised_24_records.csv",
        index=False
    )

print("\n========== VALIDATION SUMMARY ==========")
print("Expected records:", 24)
print("Extracted records:", len(df))
print("Errors:", len(errors))

if not df.empty:
    print("\nTreatment coverage:")
    print(df.groupby("treatment").size())

    print("\nShift and seed coverage:")
    print(df.groupby(["treatment", "shift"]).seed.apply(list))

if errors:
    print("\nERRORS:")
    for name, error in errors:
        print(name, ":", error)

if len(df) != 24 or errors:
    raise SystemExit("VALIDATION FAILED. Do not integrate yet.")

print("\nSUCCESS: All 24 revised Walker2d records extracted.")
print("Output:", OUT.resolve())
