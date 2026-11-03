import os
import re
import pandas as pd
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "."
TAG = "test/target_normalized_score"

algorithms = [
    "EA_NTC_NORMALIZED",
    "EA_NTC_MEAN",
    "BC_SAC",
    "BC_PAR",
    "RLPD",
    "SAC_TARGET_ONLY"
]

runs = defaultdict(list)

skip = {".git", ".venv", "node_modules", "__pycache__"}

for current, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in skip]

    for filename in files:
        if not filename.startswith("events.out.tfevents"):
            continue

        path = os.path.join(current, filename)
        p = path.lower().replace("\\", "/")

        # Identify algorithm
        if "experiments_ea_ntc_10k_96/normalized_" in p:
            method = "EA_NTC_NORMALIZED"
        elif "experiments_ea_ntc_10k_96/unnormalized_" in p:
            method = "EA_NTC_MEAN"
        elif "sac_target_only" in p:
            method = "SAC_TARGET_ONLY"
        elif "bc_sac" in p:
            method = "BC_SAC"
        elif "bc_par" in p:
            method = "BC_PAR"
        elif "rlpd" in p:
            method = "RLPD"
        else:
            continue

        env_match = re.search(
            r"(ant|halfcheetah|hopper|walker2d)-friction",
            p
        )

        shift_match = re.search(
            r"friction-srcdatatype-medium-([0-9.]+)",
            p
        )

        if not shift_match:
            shift_match = re.search(
                r"shift[_-]([0-9.]+)",
                p
            )

        seed_match = re.search(r"/r([0-2])/", p)

        if not (env_match and shift_match and seed_match):
            continue

        env = env_match.group(1)
        shift = float(shift_match.group(1))
        seed = int(seed_match.group(1))

        key = (method, env, shift, seed)

        try:
            ea = EventAccumulator(path)
            ea.Reload()

            tags = ea.Tags().get("scalars", [])

            if TAG not in tags:
                continue

            events = ea.Scalars(TAG)

            for e in events:
                if 5000 <= e.step <= 10000:
                    runs[key].append({
                        "step": e.step,
                        "value": e.value,
                        "file": path
                    })

        except Exception as error:
            print("READ ERROR:", path, error)

rows = []

for key, events in runs.items():
    method, env, shift, seed = key

    steps = [e["step"] for e in events]
    unique_steps = sorted(set(steps))

    rows.append({
        "method": method,
        "environment": env,
        "shift": shift,
        "seed": seed,
        "records": len(events),
        "unique_steps": len(unique_steps),
        "has_5000": 5000 in unique_steps,
        "has_10000": 10000 in unique_steps,
        "duplicate_steps": len(steps) - len(unique_steps),
        "complete": all(s in unique_steps for s in
                        [5000, 6000, 7000, 8000, 9000, 10000])
    })

df = pd.DataFrame(rows)

out = "RQ3_ALL_ALGORITHM_10K_AUDIT"
os.makedirs(out, exist_ok=True)

df.to_csv(
    os.path.join(out, "all_algorithm_10k_audit.csv"),
    index=False
)

print("\n========== 10K ALGORITHM INVENTORY ==========")

if not df.empty:
    summary = df.groupby("method").agg(
        runs=("seed", "count"),
        complete=("complete", "sum"),
        duplicate_steps=("duplicate_steps", "sum")
    )

    print(summary.to_string())

    print("\n========== ENVIRONMENT COVERAGE ==========")
    print(
        df.groupby(["method", "environment"])
        .size().to_string()
    )

    print("\n========== INCOMPLETE RUNS ==========")
    print(df[~df["complete"]].to_string(index=False))

    print("\n========== DUPLICATE STEP RECORDS ==========")
    print(df[df["duplicate_steps"] > 0].to_string(index=False))
else:
    print("No matching evaluation records found.")

print("\nAudit saved to:", out)
