import re
import pandas as pd
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("logs_hopper_corrected_gate_48runs")
OUT = Path("matched_final_5000/corrected_hopper_validation")
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED = [1000, 2000, 3000, 4000, 5000]
TAG = "test/target_return"

records = []
audit = []

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue

    match = re.match(
        r"(NORMALIZED|UNNORMALIZED)_hopper_shift_([\d.]+)_seed_(\d+)",
        folder.name
    )

    if not match:
        continue

    treatment, shift, seed = match.groups()
    shift = float(shift)
    seed = int(seed)

    values = {}

    event_files = list(folder.rglob("events.out.tfevents.*"))

    for event_file in event_files:
        try:
            acc = EventAccumulator(
                str(event_file),
                size_guidance={"scalars": 0}
            )
            acc.Reload()

            if TAG not in acc.Tags().get("scalars", []):
                continue

            for event in acc.Scalars(TAG):
                if event.step in EXPECTED:
                    values[event.step] = event.value

        except Exception as exc:
            print("EVENT ERROR:", event_file, exc)

    missing = [step for step in EXPECTED if step not in values]

    audit.append({
        "run": folder.name,
        "treatment": treatment,
        "shift": shift,
        "seed": seed,
        "event_files": len(event_files),
        "evaluations_found": len(values),
        "missing_steps": ",".join(map(str, missing)),
        "has_5000": 5000 in values
    })

    if seed in [0, 1, 2]:
        row = {
            "treatment": treatment,
            "reduction": (
                "normalized" if treatment == "NORMALIZED" else "mean"
            ),
            "environment": "hopper",
            "shift": shift,
            "seed": seed,
        }

        for step in EXPECTED:
            row[f"eval_{step}"] = values.get(step)

        records.append(row)

audit_df = pd.DataFrame(audit)
records_df = pd.DataFrame(records)

audit_df.to_csv(OUT / "hopper_all48_event_audit.csv", index=False)
records_df.to_csv(OUT / "hopper_corrected_24_seed012.csv", index=False)

print("\n========== TENSORBOARD VALIDATION ==========")
print("Folders audited:", len(audit_df))
print("Folders with all five evaluations:",
      (audit_df.evaluations_found == 5).sum())
print("Folders missing evaluations:",
      (audit_df.evaluations_found != 5).sum())

print("\n========== SEEDS 0, 1, 2 ==========")
print("Expected records: 24")
print("Extracted records:", len(records_df))
print("Missing evaluation values:",
      records_df[[f"eval_{s}" for s in EXPECTED]].isna().sum().sum())

print("\n========== EXTRACTED RESULTS ==========")
print(records_df.to_string(index=False))

print("\nSaved to:", OUT.resolve())
