import pandas as pd
import shutil
from pathlib import Path

MAIN = Path("matched_final_5000/matched_final_288_long.csv")
CORRECTED = Path(
    "matched_final_5000/corrected_hopper_validation/"
    "hopper_corrected_24_seed012.csv"
)

BACKUP = MAIN.with_name("matched_final_288_long_before_hopper_correction.csv")

df = pd.read_csv(MAIN)
src = pd.read_csv(CORRECTED)

eval_cols = [f"eval_{s}" for s in [1000, 2000, 3000, 4000, 5000]]

assert len(df) == 288, f"Unexpected main dataset size: {len(df)}"
assert len(src) == 24, f"Unexpected corrected dataset size: {len(src)}"

updated = 0

for _, r in src.iterrows():

    method = (
        "EA_NTC_NORMALIZED"
        if r["treatment"] == "NORMALIZED"
        else "EA_NTC_UNNORMALIZED"
    )

    mask = (
        df["method"].eq(method)
        & df["environment"].eq("hopper")
        & (pd.to_numeric(df["shift"]) - float(r["shift"])).abs().lt(1e-9)
        & df["seed"].eq(int(r["seed"]))
    )

    if mask.sum() != 1:
        raise RuntimeError(
            f"Expected exactly one match: {method}, "
            f"shift={r['shift']}, seed={r['seed']}; "
            f"found {mask.sum()}"
        )

    for col in eval_cols:
        df.loc[mask, col] = r[col]

    source_folder = (
        f"logs_hopper_corrected_gate_48runs/"
        f"{r['treatment']}_hopper_shift_{r['shift']:g}_seed_{int(r['seed'])}"
    )

    df.loc[mask, "path"] = source_folder
    df.loc[mask, "log_path"] = source_folder
    updated += 1

# Verify all corrected values are present.
assert updated == 24
assert not df[eval_cols].isna().any().any()

# Preserve original dataset.
shutil.copy2(MAIN, BACKUP)

# Save corrected dataset.
df.to_csv(MAIN, index=False)

print("\n========== HOPPER INTEGRATION ==========")
print("Updated records:", updated)
print("Unchanged records:", len(df) - updated)
print("Total records:", len(df))
print("Backup:", BACKUP.resolve())
print("Updated dataset:", MAIN.resolve())

print("\n========== CORRECTED HOPPER MEANS ==========")

subset = df[
    (df.environment == "hopper")
    & df.method.isin([
        "EA_NTC_NORMALIZED",
        "EA_NTC_UNNORMALIZED"
    ])
]

print(
    subset.groupby(["shift", "method"])[eval_cols]
    .mean()
    .round(3)
    .to_string()
)

print("\nSUCCESS: Corrected Hopper records integrated.")
