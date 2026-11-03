from pathlib import Path
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ============================================================
# EA-NTC GATE ACTIVATION DIAGNOSTICS
# ============================================================

LOG_ROOTS = [
    Path("logs_walker2d_revised_gate_48runs"),
    Path("logs_hopper_corrected_gate_48runs"),
]

OUTPUT_DIR = Path("analysis_correct_runs_4env") / "gate_activation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAGS = {
    "gate_ready": "ea_ntc/gate_ready",
    "source_weight": "ea_ntc/mean_source_weight",
    "gate_strength": "ea_ntc/adaptive_gate_strength",
    "bc_coefficient": "ea_ntc/source_bc_coefficient",
    "dynamics_updates": "ea_ntc/dynamics_updates",
    "relative_mse": "ea_ntc/heldout_relative_mse",
    "raw_source_weight": "ea_ntc/raw_mean_source_weight",
}

# Only inspect the two target failure conditions.
TARGETS = {
    "walker2d": 2.0,
    "hopper": 2.0,
}

TOL = 1e-6


def find_event_files(root):
    if not root.exists():
        print(f"WARNING: Log root not found: {root}")
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and "tfevents" in p.name
    )


def identify_run(path):
    name = str(path).lower().replace("\\", "/")

    environment = None
    for env in TARGETS:
        if env in name:
            environment = env
            break

    if environment is None:
        return None

    # Match exact shift tokens rather than loose substring matches.
    import re
    match = re.search(
        r"(?:shift_level|shift|friction)[=_-]?2(?:\.0)?(?=[^0-9.]|$)",
        name
    )

    if not match:
        return None

    seed_match = re.search(
        r"seed[=_-]?(\d+)",
        name
    )

    seed = int(seed_match.group(1)) if seed_match else None

    variant = None
    if "unnormalized" in name or "reduction_mean" in name:
        variant = "UNNORMALIZED"
    elif "normalized" in name or "reduction_normalized" in name:
        variant = "NORMALIZED"
    elif re.search(r"(?:^|[/_-])mean(?:[/_-]|$)", name):
        variant = "UNNORMALIZED"
    elif re.search(r"(?:^|[/_-])norm(?:[/_-]|$)", name):
        variant = "NORMALIZED"

    return environment, seed, variant


def read_event_file(path):
    try:
        accumulator = EventAccumulator(
            str(path),
            size_guidance={"scalars": 0, "tensors": 0}
        )
        accumulator.Reload()
    except Exception as exc:
        print(f"ERROR reading {path}: {exc}")
        return pd.DataFrame()

    available = set(accumulator.Tags().get("scalars", []))
    records = []

    for name, tag in TAGS.items():
        if tag not in available:
            continue

        for event in accumulator.Scalars(tag):
            records.append({
                "step": int(event.step),
                "metric": name,
                "value": float(event.value),
                "wall_time": float(event.wall_time),
            })

    return pd.DataFrame(records)


def first_step(df, metric, condition):
    subset = df[df["metric"] == metric].sort_values("step")

    if subset.empty:
        return np.nan

    valid = subset[condition(subset["value"].to_numpy())]

    if valid.empty:
        return np.nan

    return int(valid.iloc[0]["step"])


all_records = []
summaries = []
seen_files = set()

for root in LOG_ROOTS:
    print(f"\nSearching {root}")

    event_files = find_event_files(root)
    print(f"Found {len(event_files)} event files")

    for event_path in event_files:
        run = identify_run(event_path)

        if run is None:
            continue

        environment, seed, variant = run

        if environment not in TARGETS:
            continue

        key = str(event_path.resolve())
        if key in seen_files:
            continue
        seen_files.add(key)

        data = read_event_file(event_path)

        if data.empty:
            continue

        data["environment"] = environment
        data["seed"] = seed
        data["variant"] = variant
        data["event_file"] = str(event_path)

        all_records.append(data)

        summary = {
            "environment": environment,
            "seed": seed,
            "variant": variant,
            "event_file": str(event_path),
            "first_gate_ready": first_step(
                data,
                "gate_ready",
                lambda x: x > 0.5
            ),
            "first_nonunit_weight": first_step(
                data,
                "source_weight",
                lambda x: np.abs(x - 1.0) > TOL
            ),
            "first_positive_strength": first_step(
                data,
                "gate_strength",
                lambda x: x > TOL
            ),
            "first_positive_bc_coefficient": first_step(
                data,
                "bc_coefficient",
                lambda x: x > TOL
            ),
        }

        summaries.append(summary)


if not all_records:
    raise SystemExit(
        "\nNo matching TensorBoard runs found. "
        "Check the folder names and run naming conventions."
    )

history = pd.concat(all_records, ignore_index=True)

# If event files contain repeated entries at the same step,
# retain the latest wall-time observation.
history = (
    history.sort_values("wall_time")
    .drop_duplicates(
        ["event_file", "metric", "step"],
        keep="last"
    )
)

summary_df = pd.DataFrame(summaries)

history.to_csv(
    OUTPUT_DIR / "gate_history.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR / "activation_summary.csv",
    index=False
)

print("\n========== ACTIVATION SUMMARY ==========")
print(
    summary_df[
        [
            "environment",
            "seed",
            "variant",
            "first_gate_ready",
            "first_nonunit_weight",
            "first_positive_strength",
            "first_positive_bc_coefficient",
        ]
    ].sort_values(
        ["environment", "seed", "variant"]
    ).to_string(index=False)
)

print("\n========== AGGREGATE ACTIVATION ==========")

fields = [
    "first_gate_ready",
    "first_nonunit_weight",
    "first_positive_strength",
    "first_positive_bc_coefficient",
]

print(
    summary_df.groupby(
        ["environment", "variant"],
        dropna=False
    )[fields].agg(["count", "min", "median", "max"]).to_string()
)

print("\n========== METRIC AVAILABILITY ==========")

print(
    history.groupby(
        ["environment", "variant", "metric"],
        dropna=False
    )["step"]
    .agg(["count", "min", "max"])
    .to_string()
)

print("\nSaved files:")
print(OUTPUT_DIR / "gate_history.csv")
print(OUTPUT_DIR / "activation_summary.csv")
