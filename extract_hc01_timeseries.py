import os
import re
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOTS = [
    "logs_ea_ntc_norm_vs_mean_4env_6shift_6seed_5000",
    "experiments_ea_ntc_10k_96"
]

PATTERN = re.compile(
    r"(NORMALIZED|UNNORMALIZED)_"
    r"(halfcheetah)_shift_(0\.1)_seed_(\d+)"
)

OUTPUT = "ea_ntc_halfcheetah_01_timeseries.csv"

SELECTED_TAGS = [
    "test/target_return",
    "test/target_normalized_score",
    "test/source_return",
    "ea_ntc/mean_source_weight",
    "ea_ntc/source_weight_ess",
    "ea_ntc/source_weight_below_0_5",
    "ea_ntc/adaptive_gate_strength",
    "ea_ntc/gate_ready",
    "ea_ntc/heldout_target_mse",
    "ea_ntc/heldout_zero_delta_mse",
    "ea_ntc/heldout_relative_mse",
    "ea_ntc/mean_uncertainty",
    "ea_ntc/mean_uncertainty_penalty",
    "ea_ntc/raw_mean_source_weight",
    "train/q1"
]

rows = []
runs = set()

for root in ROOTS:
    if not os.path.isdir(root):
        continue

    for current, dirs, files in os.walk(root):
        match = PATTERN.search(current)

        if not match:
            continue

        variant, env, shift, seed = match.groups()
        runs.add((root, variant, seed))

        for filename in files:
            if not filename.startswith("events.out.tfevents"):
                continue

            path = os.path.join(current, filename)

            try:
                ea = EventAccumulator(path)
                ea.Reload()

                available = ea.Tags().get("scalars", [])

                for tag in SELECTED_TAGS:
                    if tag not in available:
                        continue

                    for event in ea.Scalars(tag):
                        rows.append({
                            "experiment": root,
                            "variant": variant,
                            "environment": env,
                            "shift": shift,
                            "seed": seed,
                            "step": event.step,
                            "metric": tag,
                            "value": event.value
                        })

            except Exception as exc:
                print("ERROR:", path, exc)

fields = [
    "experiment", "variant", "environment",
    "shift", "seed", "step", "metric", "value"
]

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("\n========== HALF CHEETAH 0.1 EXTRACTION ==========")
print("Runs:", len(runs))
print("Time-series observations:", len(rows))
print("Output:", OUTPUT)

for root, variant, seed in sorted(runs):
    print(root, variant, "seed", seed)
