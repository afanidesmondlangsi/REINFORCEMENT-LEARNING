from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import csv

root = Path("logs_ea_ntc_norm_vs_mean_4env_6shift_6seed_5000")
rows = []

for treatment in ("NORMALIZED", "UNNORMALIZED"):
    for shift in ("0.1", "0.5", "2.0", "5.0"):
        for seed in range(6):
            run = root / f"{treatment}_walker2d_shift_{shift}_seed_{seed}"
            events = list(run.rglob("events.out.tfevents.*"))

            row = {
                "treatment": treatment,
                "shift": shift,
                "seed": seed,
                "gate_ever_ready": "",
                "min_relative_mse": "",
                "min_mean_source_weight": "",
                "max_mean_source_weight": "",
                "status": "MISSING_EVENTS",
            }

            if events:
                try:
                    ea = EventAccumulator(
                        str(events[0].parent),
                        size_guidance={"scalars": 0},
                    )
                    ea.Reload()
                    tags = set(ea.Tags().get("scalars", []))

                    required = [
                        "ea_ntc/gate_ready",
                        "ea_ntc/heldout_relative_mse",
                        "ea_ntc/mean_source_weight",
                    ]

                    if all(tag in tags for tag in required):
                        gate = [v.value for v in ea.Scalars(required[0])]
                        error = [v.value for v in ea.Scalars(required[1])]
                        weight = [v.value for v in ea.Scalars(required[2])]

                        if gate and error and weight:
                            row["gate_ever_ready"] = any(v > 0.5 for v in gate)
                            row["min_relative_mse"] = min(error)
                            row["min_mean_source_weight"] = min(weight)
                            row["max_mean_source_weight"] = max(weight)
                            row["status"] = "OK"
                        else:
                            row["status"] = "EMPTY_SCALARS"
                    else:
                        row["status"] = "MISSING_TAGS"

                except Exception as exc:
                    row["status"] = f"ERROR: {exc}"

            rows.append(row)

output = Path("walker2d_gate_audit.csv")

with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(
        row["treatment"],
        row["shift"],
        row["seed"],
        "gate_ready =", row["gate_ever_ready"],
        "min_error =", row["min_relative_mse"],
        "status =", row["status"],
    )

print("\nSaved:", output)
print("Total runs:", len(rows))
print("Gate activated:", sum(
    row["gate_ever_ready"] is True for row in rows
))
