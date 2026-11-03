import os
import re
import csv
import math
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOTS = [
    "logs_ea_ntc_norm_vs_mean_4env_6shift_6seed_5000",
    "experiments_ea_ntc_10k_96",
    "logs_ea_ntc_phase15_friction_01"
]

pattern = re.compile(
    r"(NORMALIZED|UNNORMALIZED)_(ant|halfcheetah|hopper|walker2d)_shift_([0-9.]+)_seed_(\d+)"
)

metric_rows = []
run_rows = []
coverage = defaultdict(set)
tag_counts = defaultdict(int)

for root in ROOTS:
    if not os.path.isdir(root):
        print("MISSING DIRECTORY:", root)
        continue

    for current, dirs, files in os.walk(root):
        for filename in files:
            if not filename.startswith("events.out.tfevents"):
                continue

            path = os.path.join(current, filename)
            size = os.path.getsize(path)

            match = pattern.search(path)
            variant, env, shift, seed = match.groups() if match else (
                "OTHER", "OTHER", "OTHER", "OTHER"
            )

            if match:
                coverage[(root, variant, env)].add((shift, seed))

            record = {
                "root": root,
                "variant": variant,
                "environment": env,
                "shift": shift,
                "seed": seed,
                "event_file": path,
                "size_bytes": size,
                "valid": False,
                "max_step": 0,
                "metric_count": 0
            }

            if size == 0:
                record["status"] = "EMPTY"
                run_rows.append(record)
                continue

            try:
                ea = EventAccumulator(path)
                ea.Reload()

                tags = ea.Tags().get("scalars", [])
                record["valid"] = True
                record["metric_count"] = len(tags)
                record["status"] = "READABLE"

                for tag in tags:
                    events = ea.Scalars(tag)
                    if not events:
                        continue

                    values = [e.value for e in events]
                    steps = [e.step for e in events]

                    values = [v for v in values if math.isfinite(v)]

                    if not values:
                        continue

                    record["max_step"] = max(
                        record["max_step"], max(steps)
                    )

                    tag_counts[(root, tag)] += 1

                    metric_rows.append({
                        "root": root,
                        "variant": variant,
                        "environment": env,
                        "shift": shift,
                        "seed": seed,
                        "metric": tag,
                        "count": len(events),
                        "first_step": min(steps),
                        "last_step": max(steps),
                        "first_value": events[0].value,
                        "last_value": events[-1].value,
                        "minimum": min(values),
                        "maximum": max(values),
                        "mean": sum(values) / len(values)
                    })

            except Exception as exc:
                record["status"] = "ERROR: " + str(exc)

            run_rows.append(record)

print("\n========== EXPERIMENT COVERAGE ==========")

for key, runs in sorted(coverage.items()):
    root, variant, env = key
    shifts = sorted({s for s, seed in runs}, key=float)
    seeds = sorted({seed for s, seed in runs}, key=int)

    print(
        f"{root} | {variant} | {env} | "
        f"shifts={shifts} | seeds={seeds} | "
        f"runs={len(runs)}"
    )

with open("ea_ntc_metric_inventory.csv", "w", newline="") as f:
    if metric_rows:
        writer = csv.DictWriter(f, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)

with open("ea_ntc_run_inventory.csv", "w", newline="") as f:
    if run_rows:
        writer = csv.DictWriter(f, fieldnames=run_rows[0].keys())
        writer.writeheader()
        writer.writerows(run_rows)

print("\n========== EXTRACTION SUMMARY ==========")
print("Event files:", len(run_rows))
print("Readable:", sum(r["valid"] for r in run_rows))
print("Empty:", sum(r["status"] == "EMPTY" for r in run_rows))
print("Errors:", sum(r["status"] == "ERROR" for r in run_rows))
print("Metric observations:", len(metric_rows))
print("Metric CSV: ea_ntc_metric_inventory.csv")
print("Run CSV: ea_ntc_run_inventory.csv")
