import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = "experiments_ea_ntc_10k_96"
tag = "test/target_normalized_score"

rows = []

for current, dirs, files in os.walk(root):
    for file in files:
        if not file.startswith("events.out.tfevents"):
            continue

        path = os.path.join(current, file)

        try:
            ea = EventAccumulator(path)
            ea.Reload()

            if tag not in ea.Tags().get("scalars", []):
                continue

            events = ea.Scalars(tag)
            steps = [e.step for e in events]

            selected = [
                e for e in events
                if 5000 <= e.step <= 10000
            ]

            rows.append({
                "run": os.path.basename(current),
                "event_file": file,
                "records": len(events),
                "first_step": min(steps) if steps else None,
                "last_step": max(steps) if steps else None,
                "interval_records": len(selected),
                "has_5000": any(e.step == 5000 for e in events),
                "has_10000": any(e.step == 10000 for e in events),
                "duplicates": len(steps) - len(set(steps))
            })

        except Exception as e:
            print("ERROR:", path, e)

df = pd.DataFrame(rows)

if not df.empty:
    df.to_csv(
        "RQ3_test_normalized_evaluation_audit.csv",
        index=False
    )

    print("\n========== RQ3 TEST NORMALIZED AUDIT ==========")
    print(df.to_string(index=False))

    print("\n========== SUMMARY ==========")
    print("Event series:", len(df))
    print("Reaching 10K:", df["has_10000"].sum())
    print("Containing 5K:", df["has_5000"].sum())
    print("Duplicate entries:", df["duplicates"].sum())
    print("Records in interval:", df["interval_records"].sum())

    print("\nAudit saved: RQ3_test_normalized_evaluation_audit.csv")
else:
    print("No test normalized-score records found.")
