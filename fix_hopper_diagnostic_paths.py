import pandas as pd
from pathlib import Path
from datetime import datetime

BASE = Path("matched_final_5000")
MAIN = BASE / "matched_final_288_long.csv"

d = pd.read_csv(MAIN)

print("\n========== REPAIRING HOPPER PATHS ==========")

backup = BASE / (
    "matched_final_288_long_before_hopper_path_fix_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".csv"
)

mask = (
    (d["environment"] == "hopper") &
    (d["method"].isin([
        "EA_NTC_NORMALIZED",
        "EA_NTC_UNNORMALIZED"
    ])) &
    (d["shift"].isin([2.0, 5.0]))
)

rows = d.loc[mask]

assert len(rows) == 12, f"Expected 12 records, found {len(rows)}"

updates = []

for idx, row in rows.iterrows():

    treatment = (
        "NORMALIZED"
        if row["method"] == "EA_NTC_NORMALIZED"
        else "UNNORMALIZED"
    )

    shift = float(row["shift"])
    seed = int(row["seed"])

    folder_name = (
        f"{treatment}_hopper_shift_{shift:.1f}_seed_{seed}"
    )

    folder = Path("logs_hopper_corrected_gate_48runs") / folder_name

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Expected directory not found: {folder}"
        )

    old_path = row["path"]
    new_path = str(folder).replace("/", "\\")

    d.at[idx, "path"] = new_path

    if "path_norm" in d.columns:
        d.at[idx, "path_norm"] = new_path.lower()

    updates.append({
        "method": row["method"],
        "shift": shift,
        "seed": seed,
        "old_path": old_path,
        "new_path": new_path
    })

# Save backup before modifying original
d_original = pd.read_csv(MAIN)
d_original.to_csv(backup, index=False)

d.to_csv(MAIN, index=False)

print("\nUPDATED PATHS:")

for x in updates:
    print(
        f"{x['method']} | shift={x['shift']} | "
        f"seed={x['seed']}\n"
        f"OLD: {x['old_path']}\n"
        f"NEW: {x['new_path']}\n"
    )

print("Paths corrected:", len(updates))
print("Backup:", backup.resolve())
print("SUCCESS: All 12 paths corrected.")
