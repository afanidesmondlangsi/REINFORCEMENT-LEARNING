
import pandas as pd
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ============================================================
# FINAL FOUR-ENVIRONMENT, SIX-ALGORITHM, 5000-STEP ANALYSIS
# ============================================================

ROOT = Path.cwd()
AUDIT = ROOT / "full_5000_audit" / "all_candidates.csv"
OUT = ROOT / "final_288_analysis"
OUT.mkdir(exist_ok=True)

ENVS = ["hopper", "halfcheetah", "walker2d", "ant"]
SHIFTS = [0.1, 0.5, 2.0, 5.0]
SEEDS = [0, 1, 2]

METHODS = [
    "SAC_TARGET_ONLY",
    "BC_SAC",
    "RLPD",
    "BC_PAR",
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED"
]

STEPS = [1000, 2000, 3000, 4000, 5000]

RAW_TAGS = [
    "test/target_return",
    "test/target return"
]

NORM_TAGS = [
    "test/target_normalized_score",
    "test/target normalized score"
]

if not AUDIT.exists():
    raise FileNotFoundError(AUDIT)

df = pd.read_csv(AUDIT)

def norm_path(p):
    return str(p).replace("/", "\\").lower()

df["path_norm"] = df["path"].apply(norm_path)
df["shift"] = pd.to_numeric(df["shift"])
df["seed"] = pd.to_numeric(df["seed"]).astype(int)

# ============================================================
# APPROVED SOURCE SELECTION
# ============================================================

def approved_source(row):

    method = row["method"]
    env = row["environment"]
    shift = float(row["shift"])
    seed = int(row["seed"])
    path = row["path_norm"]

    s = f"{shift:.1f}"

    base = "experiments_10k_576\\results\\"

    if method == "BC_PAR":
        prefix = (
            base + "bc_par\\bc_par\\"
            f"{env}-friction-srcdatatype-medium-{s}\\r{seed}\\tb"
        )
        return path.startswith(prefix)

    if method == "BC_SAC":
        prefix = (
            base + "bc_sac\\bc_sac\\"
            f"{env}-friction-srcdatatype-medium-{s}\\r{seed}\\tb"
        )
        return path.startswith(prefix)

    if method == "RLPD":
        prefix = (
            base + "rlpd\\rlpd\\"
            f"{env}-friction-srcdatatype-medium-{s}\\r{seed}\\tb"
        )
        return path.startswith(prefix)

    if method == "SAC_TARGET_ONLY":
        prefix = (
            base + "sac_target_only\\sac_target_only\\"
            f"{env}-friction-{s}\\r{seed}\\tb"
        )
        return path.startswith(prefix)

    if method.startswith("EA_NTC_"):

        variant = (
            "NORMALIZED"
            if method == "EA_NTC_NORMALIZED"
            else "UNNORMALIZED"
        )

        if env == "hopper":
            root = "logs_hopper_corrected_gate_48runs"

        elif env == "walker2d":
            root = "logs_walker2d_revised_gate_48runs"

        else:
            root = "logs_ea_ntc_norm_vs_mean_4env_6shift_6seed_5000"

        prefix = (
            f"{root}\\{variant}_{env}_shift_{s}_seed_{seed}\\"
        )

        if not path.startswith(prefix):
            return False

        reduction = str(row["reduction"]).lower()

        if method == "EA_NTC_NORMALIZED":
            return reduction in ("normalized", "not_recorded", "")

        return reduction in ("mean", "unnormalized", "not_recorded", "")

    return False


df = df[
    df["environment"].isin(ENVS)
    & df["method"].isin(METHODS)
    & df["shift"].isin(SHIFTS)
    & df["seed"].isin(SEEDS)
    & df["complete_5000"].astype(str).str.lower().eq("true")
    & df["raw_return_available"].astype(str).str.lower().isin(
        ["true", "1"]
    )
].copy()

selected = df[df.apply(approved_source, axis=1)].copy()

keys = ["environment", "shift", "seed"]
group_keys = keys + ["method"]

# ============================================================
# STRICT UNIQUENESS VALIDATION
# ============================================================

counts = selected.groupby(group_keys).size()

expected = pd.MultiIndex.from_product(
    [ENVS, SHIFTS, SEEDS, METHODS],
    names=group_keys
)

counts = counts.reindex(expected, fill_value=0)

missing = counts[counts == 0]
duplicates = counts[counts > 1]

print("\n========== FINAL SOURCE VALIDATION ==========")
print("Approved candidate records:", len(selected))
print("Expected records:", 288)
print("Missing slots:", len(missing))
print("Duplicate slots:", len(duplicates))

if len(missing):
    missing.rename("count").reset_index().to_csv(
        OUT / "missing_slots.csv", index=False
    )
    print("\nMISSING:")
    print(missing.to_string())
    
if len(duplicates):
    duplicate_rows = selected.merge(
        duplicates.rename("count").reset_index()[group_keys],
        on=group_keys
    )
    duplicate_rows.to_csv(
        OUT / "duplicate_source_review.csv", index=False
    )
    print("\nAMBIGUOUS SOURCES:")
    print(duplicate_rows[group_keys + ["path"]].to_string(index=False))

if len(missing) or len(duplicates):
    raise RuntimeError(
        "Selection is not uniquely matched. No final analysis performed."
    )

if len(selected) != 288:
    raise RuntimeError("Expected exactly 288 selected records.")

# ============================================================
# READ BOTH RAW AND NORMALIZED TENSORBOARD METRICS
# ============================================================

def read_tag(acc, candidates):

    tags = set(acc.Tags().get("scalars", []))

    tag = next(
        (t for t in candidates if t in tags),
        None
    )

    if tag is None:
        return {}

    values = {}

    for event in acc.Scalars(tag):
        values[int(event.step)] = float(event.value)

    return values


records = []
errors = []

for _, row in selected.iterrows():

    tb = ROOT / row["path"]

    try:
        acc = EventAccumulator(
            str(tb),
            size_guidance={"scalars": 0}
        )
        acc.Reload()

        raw = read_tag(acc, RAW_TAGS)
        norm = read_tag(acc, NORM_TAGS)

        if not all(s in raw for s in STEPS):
            raise ValueError("Incomplete raw-return evaluations")

        if not all(s in norm for s in STEPS):
            raise ValueError("Incomplete normalized-score evaluations")

        raw_auc = np.trapz(
            [raw[s] for s in STEPS],
            x=STEPS
        )

        norm_auc = np.trapz(
            [norm[s] for s in STEPS],
            x=STEPS
        )

        record = {
            "environment": row["environment"],
            "shift": float(row["shift"]),
            "seed": int(row["seed"]),
            "method": row["method"],
            "source_path": row["path"],
            "auc_return_1000_5000": raw_auc,
            "auc_normalized_1000_5000": norm_auc,
            "adaptation_gain_1000_5000": (
                norm[5000] - norm[1000]
            )
        }

        for step in STEPS:
            record[f"target_return_{step}"] = raw[step]
            record[f"normalized_score_{step}"] = norm[step]

        records.append(record)

    except Exception as exc:
        errors.append({
            "path": row["path"],
            "method": row["method"],
            "environment": row["environment"],
            "shift": row["shift"],
            "seed": row["seed"],
            "error": str(exc)
        })

if errors:
    pd.DataFrame(errors).to_csv(
        OUT / "metric_read_errors.csv", index=False
    )
    raise RuntimeError(
        f"{len(errors)} runs failed metric validation. "
        "Analysis stopped."
    )

long = pd.DataFrame(records)

if len(long) != 288:
    raise RuntimeError("Metric dataset does not contain 288 records.")

long.to_csv(
    OUT / "matched_5000_long.csv",
    index=False
)

# ============================================================
# SUMMARY: MEAN AND SAMPLE STANDARD DEVIATION
# ============================================================

metric_columns = [
    "normalized_score_1000",
    "normalized_score_2000",
    "normalized_score_3000",
    "normalized_score_4000",
    "normalized_score_5000",
    "target_return_5000",
    "auc_normalized_1000_5000",
    "auc_return_1000_5000",
    "adaptation_gain_1000_5000"
]

summary = (
    long.groupby(["environment", "shift", "method"])[metric_columns]
    .agg(["mean", "std", "count"])
)

summary.to_csv(
    OUT / "matched_5000_summary.csv"
)

# ============================================================
# NEGATIVE TRANSFER: PAIRED AGAINST TARGET-ONLY SAC
# ============================================================

transfer_methods = [
    "BC_SAC",
    "RLPD",
    "BC_PAR",
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED"
]

transfer_results = []

for metric in [
    "normalized_score_1000",
    "normalized_score_3000",
    "normalized_score_5000",
    "auc_normalized_1000_5000"
]:

    wide = long.pivot(
        index=keys,
        columns="method",
        values=metric
    )

    for method in transfer_methods:

        delta = (
            wide[method] - wide["SAC_TARGET_ONLY"]
        ).dropna()

        temp = delta.rename("transfer_advantage").reset_index()
        temp["method"] = method
        temp["metric"] = metric

        transfer_results.append(temp)

transfer = pd.concat(
    transfer_results,
    ignore_index=True
)

transfer.to_csv(
    OUT / "paired_transfer_advantages.csv",
    index=False
)

transfer_summary = (
    transfer.groupby(
        ["environment", "shift", "method", "metric"]
    )["transfer_advantage"]
    .agg(
        mean="mean",
        std="std",
        n="count"
    )
    .reset_index()
)

negative_rate = (
    transfer.assign(
        negative=transfer["transfer_advantage"] < 0
    )
    .groupby(
        ["environment", "shift", "method", "metric"]
    )["negative"]
    .mean()
    .reset_index(name="negative_transfer_rate")
)

transfer_summary = transfer_summary.merge(
    negative_rate,
    on=["environment", "shift", "method", "metric"]
)

transfer_summary.to_csv(
    OUT / "negative_transfer_summary.csv",
    index=False
)

# ============================================================
# EA-NTC MEAN VS NORMALIZED
# ============================================================

ea_comparison = []

for metric in metric_columns:

    wide = long.pivot(
        index=keys,
        columns="method",
        values=metric
    )

    delta = (
        wide["EA_NTC_NORMALIZED"]
        - wide["EA_NTC_UNNORMALIZED"]
    ).rename("normalized_minus_mean").reset_index()

    delta["metric"] = metric
    ea_comparison.append(delta)

ea_comparison = pd.concat(
    ea_comparison,
    ignore_index=True
)

ea_comparison.to_csv(
    OUT / "ea_ntc_normalized_vs_mean.csv",
    index=False
)

# ============================================================
# FINAL COVERAGE REPORT
# ============================================================

print("\n========== FINAL 5000-STEP ANALYSIS ==========")
print("Metric records:", len(long))
print("Algorithms:", long["method"].nunique())
print("Environments:", long["environment"].nunique())
print("Shifts:", sorted(long["shift"].unique()))
print("Seeds:", sorted(long["seed"].unique()))

print("\nCoverage by environment:")
print(
    long.groupby("environment")
    .size()
    .to_string()
)

print("\nCoverage by algorithm:")
print(
    long.groupby("method")
    .size()
    .to_string()
)

print("\n========== ANALYSIS OUTPUTS ==========")

for file in sorted(OUT.glob("*.csv")):
    print(file.name)

print("\nSUCCESS: 288 matched records analyzed.")
print("Output directory:", OUT.resolve())
