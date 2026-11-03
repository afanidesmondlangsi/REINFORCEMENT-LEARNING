
import pandas as pd
from pathlib import Path

AUDIT = Path("full_5000_audit/all_candidates.csv")
OUT = Path("matched_corrected_revised_5000")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(AUDIT)


def norm_path(p):
    return str(p).replace("/", "\\").lower()


df["path_norm"] = df["path"].apply(norm_path)


def valid_source(row):
    m = row["method"]
    env = row["environment"]
    shift = float(row["shift"])
    seed = int(row["seed"])
    p = row["path_norm"]

    s = str(shift)

    if m == "BC_PAR":
        return p.startswith(
            f"experiments_10k_576\\results\\bc_par\\bc_par\\"
            f"{env}-friction-srcdatatype-medium-{s}\\r{seed}\\tb"
        )

    if m == "BC_SAC":
        return p.startswith(
            f"experiments_10k_576\\results\\bc_sac\\bc_sac\\"
            f"{env}-friction-srcdatatype-medium-{s}\\r{seed}\\tb"
        )

    if m.startswith("EA_NTC_"):
        variant = (
            "normalized" if m == "EA_NTC_NORMALIZED"
            else "unnormalized"
        )

        root = (
            "logs_hopper_corrected_gate_48runs"
            if env == "hopper"
            else "logs_walker2d_revised_gate_48runs"
        )

        return p.startswith(
            f"{root}\\{variant}_{env}_shift_{s}_seed_{seed}\\"
        )

    if m == "RLPD":
        if env == "hopper" and shift == 0.1:
            root = (
                "rq1_rlpd_test\\rlpd"
                if seed == 0
                else "rq1_rlpd\\rlpd"
            )
        else:
            root = "rq2_generalization\\rlpd"

        return p.startswith(root + "\\")

    if m == "SAC_TARGET_ONLY":
        if env == "hopper" and shift == 0.1:
            root = "rq1_target_only\\sac_target_only"
        else:
            root = "rq2_generalization\\sac_target_only"

        return p.startswith(root + "\\")

    return False


methods = [
    "BC_PAR",
    "BC_SAC",
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED",
    "RLPD",
    "SAC_TARGET_ONLY"
]

# Filter eligible environments, methods and completed runs.
df = df[
    df["environment"].isin(["hopper", "walker2d"])
    & df["method"].isin(methods)
    & df["complete_5000"].astype(str).str.lower().eq("true")
].copy()

# Retain only approved source directories.
selected = df[df.apply(valid_source, axis=1)].copy()

# Use seeds 0, 1 and 2 exclusively.
selected = selected[
    selected["seed"].isin([0, 1, 2])
].copy()

keys = ["environment", "shift", "seed"]

# Count eligible candidates for each method and condition.
counts = (
    selected.groupby(keys + ["method"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=methods, fill_value=0)
)

print("\n========== FINAL SOURCE AUDIT ==========")
print("Selected records:", len(selected))

print("\nCoverage:")
print(counts.to_string())

# Strict matching: exactly one candidate per method.
matched = counts[
    (counts[methods] == 1).all(axis=1)
].reset_index()

print("\nMatched conditions:", len(matched))

# Expected matched conditions.
expected = {
    "hopper": [
        (0.1, 0), (0.1, 1), (0.1, 2),
        (0.5, 0), (0.5, 1), (0.5, 2),
        (2.0, 0), (2.0, 1), (2.0, 2)
    ],
    "walker2d": [
        (0.5, 0), (0.5, 1), (0.5, 2),
        (2.0, 0), (2.0, 1), (2.0, 2)
    ]
}

expected_df = pd.DataFrame([
    {"environment": env, "shift": shift, "seed": seed}
    for env, conditions in expected.items()
    for shift, seed in conditions
])

check = expected_df.merge(
    matched[keys],
    on=keys,
    how="left",
    indicator=True
)

print("\nMissing matched conditions:")
missing = check[check["_merge"] == "left_only"][keys]
print(missing.to_string(index=False) if not missing.empty else "None")

if len(matched) != 15:
    print("\nWARNING: Expected 15 matched conditions.")
else:
    print("\nSUCCESS: All 15 conditions matched.")

# Retain only strictly matched conditions.
final = selected.merge(
    matched[keys],
    on=keys,
    how="inner"
)

final = final.sort_values(
    ["environment", "shift", "seed", "method"]
)

# Verify expected record count.
print("\n========== FINAL VALIDATION ==========")
print("Final records:", len(final))
print("Unique conditions:", len(matched))
print("Methods:", final["method"].nunique())
print("Seeds:", sorted(final["seed"].unique()))

if len(final) != 90:
    print("WARNING: Expected 90 records. Inspect coverage.")
else:
    print("SUCCESS: 90 matched records verified.")

# Save outputs.
final.to_csv(
    OUT / "matched_corrected_revised_long.csv",
    index=False
)

print("\nSaved:", OUT.resolve())

