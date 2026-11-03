import pandas as pd
from pathlib import Path
from datetime import datetime

BASE = Path("matched_final_5000")
MAIN = BASE / "matched_final_288_long.csv"

d = pd.read_csv(MAIN)

mask = (
    (d.environment == "hopper") &
    d.method.isin([
        "EA_NTC_NORMALIZED",
        "EA_NTC_UNNORMALIZED"
    ])
)

assert mask.sum() == 24

backup = BASE / (
    "before_hopper_event_path_repair_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".csv"
)

updates = []

for idx, row in d.loc[mask].iterrows():

    treatment = (
        "NORMALIZED"
        if row.method == "EA_NTC_NORMALIZED"
        else "UNNORMALIZED"
    )

    shift = float(row["shift"])
    seed = int(row["seed"])

    run_root = (
        Path("logs_hopper_corrected_gate_48runs") /
        f"{treatment}_hopper_shift_{shift:.1f}_seed_{seed}"
    )

    if not run_root.is_dir():
        raise FileNotFoundError(run_root)

    events = list(run_root.rglob("events.out.tfevents.*"))

    if not events:
        raise RuntimeError(
            f"No TensorBoard event files found: {run_root}"
        )

    parents = sorted(set(p.parent for p in events))

    if len(parents) != 1:
        raise RuntimeError(
            f"Multiple event directories found: {run_root}\n"
            f"{parents}"
        )

    event_dir = parents[0]

    d.at[idx, "path"] = str(event_dir)

    if "path_norm" in d.columns:
        d.at[idx, "path_norm"] = str(event_dir).lower()

    updates.append({
        "method": row.method,
        "shift": shift,
        "seed": seed,
        "event_files": len(events),
        "path": str(event_dir)
    })

d.to_csv(backup, index=False)

# Restore the original and write only after all 24 paths validate.
d.to_csv(MAIN, index=False)

print("\n========== HOPPER EVENT PATH REPAIR ==========")

for x in updates:
    print(
        f"{x['method']} | shift={x['shift']} | "
        f"seed={x['seed']} | event files={x['event_files']}"
    )
    print("PATH:", x["path"])

print("\nRepaired records:", len(updates))
print("Backup:", backup.resolve())
print("SUCCESS")
