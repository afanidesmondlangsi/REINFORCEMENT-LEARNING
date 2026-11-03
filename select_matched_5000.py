import json
import pandas as pd
import numpy as np

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(".")
AUDIT = ROOT / "full_5000_audit"
OUTPUT = ROOT / "matched_5000_final"

OUTPUT.mkdir(exist_ok=True)

METHODS = [
    "SAC_TARGET_ONLY",
    "BC_SAC",
    "RLPD",
    "BC_PAR",
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED",
]

KEYS = ["environment", "shift", "seed"]

STEPS = [1000, 2000, 3000, 4000, 5000]

METRICS = {
    "normalized_score": "test/target_normalized_score",
    "target_return": "test/target_return",
}


def as_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def read_events(directory):
    directory = Path(directory)

    if not directory.exists():
        return None

    accumulator = EventAccumulator(str(directory))
    accumulator.Reload()

    tags = accumulator.Tags().get("scalars", [])

    result = {}

    for metric_name, tag in METRICS.items():

        if tag not in tags:
            return None

        events = accumulator.Scalars(tag)

        values = {
            int(event.step): float(event.value)
            for event in events
        }

        if not all(step in values for step in STEPS):
            return None

        result[metric_name] = values

    return result


def canonical(path):
    normalized = str(path).replace("/", "\\").lower()

    return normalized.startswith(
        "experiments_10k_576\\"
    )


print("\n========== LOADING EXISTING AUDIT ==========")

candidates = pd.read_csv(
    AUDIT / "all_candidates.csv"
)

print("Candidate records:", len(candidates))

candidates["complete_5000"] = (
    candidates["complete_5000"].apply(as_bool)
)

candidates["raw_return_available"] = (
    candidates["raw_return_available"].apply(as_bool)
)

candidates["shift"] = pd.to_numeric(
    candidates["shift"]
)

candidates["seed"] = pd.to_numeric(
    candidates["seed"]
).astype(int)

candidates = candidates[
    candidates["method"].isin(METHODS)
    & candidates["complete_5000"]
].copy()

print("Eligible candidate records:", len(candidates))


# ============================================================
# DEDUPLICATE IDENTICAL DIRECTORY REFERENCES
# ============================================================

candidates = candidates.drop_duplicates(
    subset=KEYS + ["method", "path"]
)


# ============================================================
# SELECT USING EXPERIMENT PROVENANCE
# ============================================================

selected = []
unresolved = []
cache = {}

groups = candidates.groupby(
    ["method"] + KEYS,
    dropna=False
)

print("\n========== SELECTING EXPERIMENTS ==========")

for (method, environment, shift, seed), group in groups:

    valid = []

    for _, row in group.iterrows():

        directory = str(row["path"])

        if directory not in cache:
            cache[directory] = read_events(directory)

        event_data = cache[directory]

        if event_data is not None:
            valid.append((row, event_data))

    if not valid:
        continue

    preferred = [
        item for item in valid
        if canonical(item[0]["path"])
    ]

    if len(preferred) == 1:
        chosen = preferred[0]

    elif len(preferred) > 1:
        unresolved.append({
            "method": method,
            "environment": environment,
            "shift": shift,
            "seed": seed,
            "reason": "MULTIPLE_CANONICAL_CANDIDATES",
            "paths": " | ".join(
                str(item[0]["path"])
                for item in preferred
            ),
        })
        continue

    elif len(valid) == 1:
        chosen = valid[0]

    else:
        unresolved.append({
            "method": method,
            "environment": environment,
            "shift": shift,
            "seed": seed,
            "reason": "MULTIPLE_NONCANONICAL_CANDIDATES",
            "paths": " | ".join(
                str(item[0]["path"])
                for item in valid
            ),
        })
        continue

    row, event_data = chosen

    record = {
        "method": method,
        "environment": environment,
        "shift": shift,
        "seed": seed,
        "path": str(row["path"]),
        "config_signature": row["config_signature"],
        "reduction": row["reduction"],
    }

    for metric_name, values in event_data.items():

        for step in STEPS:
            record[
                f"{metric_name}_{step}"
            ] = values[step]

    # AUC of normalized target score over observed
    # checkpoints from 1000 to 5000.
    y = [
        event_data["normalized_score"][step]
        for step in STEPS
    ]

    record["auc_1000_5000"] = float(
        np.trapz(y, x=STEPS)
    )

    record["mean_score_1000_5000"] = (
        record["auc_1000_5000"] / 4000
    )

    selected.append(record)


# ============================================================
# SAVE SELECTED AND UNRESOLVED EXPERIMENTS
# ============================================================

selected_df = pd.DataFrame(selected)
unresolved_df = pd.DataFrame(unresolved)

selected_df.to_csv(
    OUTPUT / "selected_experiments.csv",
    index=False
)

unresolved_df.to_csv(
    OUTPUT / "unresolved_duplicates.csv",
    index=False
)


# ============================================================
# MATCH ALL SIX METHODS
# ============================================================

if selected_df.empty:
    raise RuntimeError(
        "No eligible experiments were selected."
    )

counts = selected_df.groupby(
    KEYS
)["method"].nunique()

matched_keys = counts[counts == len(METHODS)].index

matched = selected_df.set_index(
    KEYS
).loc[matched_keys].reset_index()

matched.to_csv(
    OUTPUT / "matched_5000_long.csv",
    index=False
)


# ============================================================
# CREATE WIDE COMPARISON DATASET
# ============================================================

comparison_columns = [
    "normalized_score_5000",
    "target_return_5000",
    "auc_1000_5000",
    "mean_score_1000_5000",
]

wide = matched.pivot(
    index=KEYS,
    columns="method",
    values=comparison_columns
)

wide.columns = [
    f"{metric}_{method}"
    for metric, method in wide.columns
]

wide = wide.reset_index()

wide.to_csv(
    OUTPUT / "matched_5000_wide.csv",
    index=False
)


# ============================================================
# SUMMARY: MEAN AND STANDARD DEVIATION ACROSS SEEDS
# ============================================================

summary_metrics = [
    "normalized_score_5000",
    "target_return_5000",
    "auc_1000_5000",
    "mean_score_1000_5000",
]

summary = matched.groupby(
    ["environment", "shift", "method"]
)[summary_metrics].agg(
    ["mean", "std", "count"]
)

summary.to_csv(
    OUTPUT / "matched_5000_summary.csv"
)


# ============================================================
# REPORT
# ============================================================

print("\n========== FINAL SELECTION SUMMARY ==========")

print("Selected experiments:", len(selected_df))
print("Unresolved duplicate groups:", len(unresolved_df))
print("Matched six-method conditions:", len(matched_keys))

print("\nCoverage by method:")

print(
    selected_df.groupby("method").size()
)

print("\nMatched conditions by environment:")

print(
    matched.groupby("environment")[KEYS]
    .size()
)

print("\nMatched conditions by seed:")

print(
    matched.groupby("seed").size()
)

print("\nMatched conditions by shift:")

print(
    matched.groupby("shift").size()
)

print("\nOUTPUT FILES:")

for file in sorted(OUTPUT.glob("*.csv")):
    print(file)

print("\nSELECTION COMPLETE.")
