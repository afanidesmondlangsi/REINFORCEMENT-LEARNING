import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = r"experiments_10k_576\results\BC_SAC"

TARGET = "halfcheetah-friction-srcdatatype-medium-0.1"

TAGS = [
    "test/target_return",
    "test/target_normalized_score",
    "test/source_return",
    "train/target_return",
    "train/target_normalized_score"
]

rows = []
found = set()

for current, dirs, files in os.walk(ROOT):

    if TARGET not in current:
        continue

    for filename in files:

        if not filename.startswith("events.out.tfevents"):
            continue

        path = os.path.join(current, filename)

        try:
            ea = EventAccumulator(path)
            ea.Reload()

            available = ea.Tags().get("scalars", [])

            seed = next(
                (x for x in current.replace("\\", "/").split("/")
                 if x.startswith("r") and x[1:].isdigit()),
                "unknown"
            )

            found.add(seed)

            for tag in TAGS:

                if tag not in available:
                    continue

                for e in ea.Scalars(tag):
                    rows.append({
                        "algorithm": "BC_SAC",
                        "environment": "halfcheetah",
                        "shift": "0.1",
                        "seed": seed,
                        "step": e.step,
                        "metric": tag,
                        "value": e.value
                    })

        except Exception as exc:
            print("ERROR:", path, exc)

output = "bc_sac_halfcheetah_01_timeseries.csv"

with open(output, "w", newline="") as f:

    fields = [
        "algorithm", "environment", "shift",
        "seed", "step", "metric", "value"
    ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("\n========== BC-SAC EXTRACTION ==========")
print("Seeds found:", sorted(found))
print("Observations:", len(rows))
print("Output:", output)
