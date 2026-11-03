
import pandas as pd
from pathlib import Path

AUDIT = Path("full_5000_audit/all_candidates.csv")
OUT = Path("matched_final_5000")
OUT.mkdir(exist_ok=True)

METHODS = [
    "BC_PAR",
    "BC_SAC",
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED",
    "RLPD",
    "SAC_TARGET_ONLY"
]

ENVIRONMENTS = ["hopper", "halfcheetah", "walker2d", "ant"]
SHIFTS = [0.1, 0.5, 2.0, 5.0]
SEEDS = [0, 1, 2]

df = pd.read_csv(AUDIT)

df["path_norm"] = (
    df["path"].astype(str)
    .str.replace("/", "\\", regex=False)
    .str.lower()
)

df["shift"] = pd.to_numeric(df["shift"])
df["seed"] = pd.to_numeric(df["seed"]).astype(int)

df = df[
    df["environment"].isin(ENVIRONMENTS)
    & df["method"].isin(METHODS)
    & df["shift"].isin(SHIFTS)
    & df["seed"].isin(SEEDS)
    & df["complete_5000"].astype(str).str.lower().eq("true")
].copy()


def source_priority(row):
    method = row["method"]
    path = row["path_norm"]

    # BC-SAC: use consolidated experiment branch.
    if method == "BC_SAC":
        return (
            0 if path.startswith(
                "experiments_10k_576\\results\\bc_sac\\bc_sac\\"
            ) else 10
        )

    # Both EA-NTC variants: use matched normalized-vs-mean runs.
    if method in ["EA_NTC_NORMALIZED", "EA_NTC_UNNORMALIZED"]:
        return (
            0 if path.startswith(
                "logs_ea_ntc_norm_vs_mean_4env_6shift_6seed_5000\\"
            ) else 10
        )

    # Other algorithms have no competing eligible sources
    # in the current audit.
    return 0


df["priority"] = df.apply(source_priority, axis=1)

keys = ["environment", "shift", "seed", "method"]

df = df.sort_values(keys + ["priority", "path_norm"])

selected = []

for key, group in df.groupby(keys, sort=True):

    best_priority = group["priority"].min()
    best = group[group["priority"] == best_priority]

    if len(best) != 1:
        raise RuntimeError(
            f"Ambiguous preferred source for {key}: "
            f"{best['path'].tolist()}"
        )

    selected.append(best.iloc[0])

final = pd.DataFrame(selected).reset_index(drop=True)

# Expected complete experimental design.
expected = pd.MultiIndex.from_product(
    [ENVIRONMENTS, SHIFTS, SEEDS, METHODS],
    names=keys
).to_frame(index=False)

actual = final[keys]

check = expected.merge(
    actual,
    on=keys,
    how="left",
    indicator=True
)

missing = check[check["_merge"] == "left_only"]

duplicates = actual.duplicated(keys).sum()

print("\n========== FINAL SOURCE VALIDATION ==========")
print("Expected records:", len(expected))
print("Selected records:", len(final))
print("Missing configurations:", len(missing))
print("Duplicate configurations:", duplicates)

if not missing.empty:
    print("\nMISSING:")
    print(missing.to_string(index=False))

if len(final) != 288 or len(missing) or duplicates:
    raise RuntimeError(
        "Final dataset is not uniquely matched. "
        "No analysis performed."
    )

print("\nSUCCESS: All 288 records uniquely matched.")

print("\n========== METHOD COVERAGE ==========")
print(final.groupby("method").size())

print("\n========== ENVIRONMENT COVERAGE ==========")
print(final.groupby("environment").size())

print("\n========== SELECTED SOURCE DIRECTORIES ==========")
for method in METHODS:
    print(f"\n{method}")
    for path in sorted(
        final.loc[final["method"] == method, "path"].unique()
    ):
        print(path)

final = final.sort_values(
    ["environment", "shift", "seed", "method"]
)

final.to_csv(
    OUT / "matched_final_288_long.csv",
    index=False
)

final[keys + ["path"]].to_csv(
    OUT / "selected_source_manifest.csv",
    index=False
)

print("\nSaved final dataset:", OUT.resolve())
print("Saved source manifest.")
