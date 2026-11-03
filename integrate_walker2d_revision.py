import pandas as pd
import hashlib
import json
from pathlib import Path

BASE = Path("matched_final_5000")
MAIN = BASE / "matched_final_288_long.csv"
SOURCE = BASE / "walker2d_revised_validation" / "walker2d_revised_24_records.csv"

df = pd.read_csv(MAIN)
new = pd.read_csv(SOURCE)

keys = ["environment", "shift", "seed", "method"]
metrics = [f"eval_{x}" for x in [1000, 2000, 3000, 4000, 5000]]

new["method"] = new["treatment"].map({
    "NORMALIZED": "EA_NTC_NORMALIZED",
    "UNNORMALIZED": "EA_NTC_UNNORMALIZED"
})

new["environment"] = "walker2d"

assert len(df) == 288
assert len(new) == 24
assert not df.duplicated(keys).any()
assert not new.duplicated(keys).any()

assert set(new["seed"]) == {0, 1, 2}
assert set(new["shift"]) == {0.1, 0.5, 2.0, 5.0}
assert new[metrics].notna().all().all()

mask = (
    df["environment"].eq("walker2d") &
    df["method"].isin([
        "EA_NTC_NORMALIZED",
        "EA_NTC_UNNORMALIZED"
    ])
)

assert mask.sum() == 24, f"Expected 24 existing records, found {mask.sum()}"

backup = BASE / "matched_final_288_before_walker2d_revision.csv"
df.to_csv(backup, index=False)

updates = 0

for _, row in new.iterrows():

    match = (
        df["environment"].eq("walker2d") &
        df["method"].eq(row["method"]) &
        df["shift"].eq(row["shift"]) &
        df["seed"].eq(row["seed"])
    )

    assert match.sum() == 1

    idx = df.index[match][0]

    df.loc[idx, "path"] = row["path"]

    if "path_norm" in df.columns:
        df.loc[idx, "path_norm"] = str(row["path"]).replace("\\", "/")

    for metric in metrics:
        df.loc[idx, metric] = row[metric]

    df.loc[idx, "reduction"] = row["reduction"]
    df.loc[idx, "max_step"] = 5000
    df.loc[idx, "complete_5000"] = True
    df.loc[idx, "tag"] = "test/target_return"
    df.loc[idx, "steps"] = "1000;2000;3000;4000;5000"

    event_dir = Path(row["path"])
    event_files = sorted(event_dir.glob("events.out.tfevents.*"))

    assert event_files, f"No event files: {event_dir}"

    combined = hashlib.sha256()

    for file in event_files:
        combined.update(file.read_bytes())

    df.loc[idx, "event_files"] = len(event_files)
    df.loc[idx, "event_hashes"] = combined.hexdigest()

    # Avoid retaining the old experiment's configuration signature.
    run_root = Path("logs_walker2d_revised_gate_48runs") / (
        "NORMALIZED" if row["treatment"] == "NORMALIZED"
        else "UNNORMALIZED"
    ) + Path("")

    config_files = list(run_root.rglob("run_config.json"))
    config_files = [
        f for f in config_files
        if f"{row['shift']}" in str(f)
        and f"seed_{row['seed']}" in str(f)
    ]

    if config_files:
        config = json.loads(config_files[0].read_text(encoding="utf-8"))
        df.loc[idx, "config_signature"] = json.dumps(
            config, sort_keys=True
        )
    else:
        df.loc[idx, "config_signature"] = "NOT_RECORDED"

    updates += 1

assert len(df) == 288
assert not df.duplicated(keys).any()

df.to_csv(MAIN, index=False)

print("\n========== WALKER2D INTEGRATION ==========")
print("Updated records:", updates)
print("Unchanged records:", len(df) - updates)
print("Total records:", len(df))
print("Backup:", backup.resolve())
print("Updated dataset:", MAIN.resolve())

print("\n========== WALKER2D CORRECTED MEANS ==========")

w = df[
    (df.environment == "walker2d") &
    df.method.str.startswith("EA_NTC")
]

print(
    w.groupby(["shift", "method"])[metrics]
    .mean().round(3).to_string()
)

print("\nSUCCESS: Walker2d revision integrated.")
