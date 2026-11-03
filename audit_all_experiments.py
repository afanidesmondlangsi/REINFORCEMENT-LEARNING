from pathlib import Path
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import csv
import re

root = Path.cwd()
records = []
tag = "test/target_normalized_score"

for file in root.rglob("events.out.tfevents.*"):
    path = file.parent
    parts = path.relative_to(root).parts
    text = str(path).lower()

    try:
        ea = EventAccumulator(str(file), size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if tag in tags:
            values = ea.Scalars(tag)
            max_step = max(x.step for x in values)
            evaluations = len(values)
        else:
            max_step = None
            evaluations = 0
        status = "OK" if max_step is not None else "NO_SCORE"
    except Exception:
        max_step = None
        evaluations = 0
        status = "ERROR"

    env = next((x for x in ["halfcheetah", "walker2d", "hopper", "ant"] if x in text), "")
    seed_match = re.search(r"(?:seed[_-]?|[\\/]+r)(\d+)", text)
    seed = seed_match.group(1) if seed_match else ""

    records.append({
        "experiment": parts[0] if parts else "",
        "environment": env,
        "seed": seed,
        "max_step": max_step,
        "evaluations": evaluations,
        "status": status,
        "path": str(path)
    })

out = root / "all_experiment_audit.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys() if records else
        ["experiment","environment","seed","max_step","evaluations","status","path"])
    writer.writeheader()
    writer.writerows(records)

groups = defaultdict(list)
for r in records:
    groups[r["experiment"]].append(r)

print("\nEXPERIMENT AUDIT")
print(f"{'Experiment':48} {'Runs':>6} {'10k+':>6} {'5k+':>6} {'Max step':>10}")
print("-" * 85)

for name, rows in sorted(groups.items()):
    steps = [r["max_step"] for r in rows if r["max_step"] is not None]
    n10 = sum(x >= 10000 for x in steps)
    n5 = sum(x >= 5000 for x in steps)
    print(f"{name[:48]:48} {len(rows):6} {n10:6} {n5:6} {max(steps) if steps else 'N/A':>10}")

print(f"\nTOTAL EVENT FILES: {len(records)}")
print(f"FULL REPORT: {out}")
