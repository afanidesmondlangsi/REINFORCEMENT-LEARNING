from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

root = Path("logs_ea_ntc_norm_vs_mean_4env_6shift_6seed_5000")

treatments = ("NORMALIZED", "UNNORMALIZED")
seeds = range(6)
shifts = ("0.1", "0.5")

tags = [
    "test/target_normalized_score",
    "ea_ntc/heldout_relative_mse",
    "ea_ntc/gate_ready",
    "ea_ntc/mean_source_weight",
]

def load_run(treatment, shift, seed):
    name = f"{treatment}_hopper_shift_{shift}_seed_{seed}"
    directory = root / name

    files = list(directory.rglob("events.out.tfevents.*"))

    if len(files) != 1:
        raise RuntimeError(
            f"{name}: expected one event file, found {len(files)}"
        )

    ea = EventAccumulator(
        str(files[0].parent),
        size_guidance={"scalars": 0},
    )
    ea.Reload()

    result = {}

    for tag in tags:
        if tag not in ea.Tags().get("scalars", []):
            result[tag] = None
        else:
            result[tag] = {
                int(v.step): float(v.value)
                for v in ea.Scalars(tag)
            }

    return result

print("=" * 75)
print("HOPPER FRICTION 0.1 VS 0.5")
print("=" * 75)

for treatment in treatments:
    print(f"\nTREATMENT: {treatment}")

    for seed in seeds:
        try:
            a = load_run(treatment, "0.1", seed)
            b = load_run(treatment, "0.5", seed)

            print(f"\nSeed {seed}")

            for tag in tags:
                x = a[tag]
                y = b[tag]

                if x is None or y is None:
                    print(f"  {tag}: MISSING")
                    continue

                common = sorted(set(x) & set(y))

                if not common:
                    print(f"  {tag}: NO COMMON STEPS")
                    continue

                differences = [
                    abs(x[step] - y[step])
                    for step in common
                ]

                exact = (
                    set(x) == set(y)
                    and all(d == 0 for d in differences)
                )

                print(
                    f"  {tag}: "
                    f"exact={exact}, "
                    f"max_difference={max(differences):.10g}, "
                    f"common_steps={len(common)}"
                )

                if tag == "test/target_normalized_score":
                    print("    0.1:", [
                        round(x[s], 6) for s in common
                    ])
                    print("    0.5:", [
                        round(y[s], 6) for s in common
                    ])

        except Exception as exc:
            print(f"Seed {seed}: ERROR: {exc}")
