from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path("logs_ea_ntc_norm_vs_mean_4env_6shift_6seed_5000")

tags_to_check = [
    "ea_ntc/gate_ready",
    "ea_ntc/mean_source_weight",
    "ea_ntc/dynamics_updates",
    "ea_ntc/adaptive_gate_strength",
    "ea_ntc/source_weight_ess",
    "test/target_normalized_score",
    "test/target_return",
]

for treatment in ("NORMALIZED", "UNNORMALIZED"):

    run = root / f"{treatment}_walker2d_shift_0.5_seed_0"

    event_files = sorted(run.rglob("events.out.tfevents.*"))

    print("\n" + "=" * 70)
    print("TREATMENT:", treatment)
    print("EVENT FILES:", len(event_files))

    if not event_files:
        print("ERROR: No event files")
        continue

    for event_file in event_files:
        print("EVENT PATH:", event_file)

        ea = EventAccumulator(
            str(event_file.parent),
            size_guidance={"scalars": 0},
        )
        ea.Reload()

        available = set(ea.Tags().get("scalars", []))

        print("AVAILABLE SCALAR TAGS:")
        for tag in sorted(available):
            print(" ", tag)

        for tag in tags_to_check:

            print("\nTAG:", tag)

            if tag not in available:
                print("NOT LOGGED")
                continue

            values = ea.Scalars(tag)

            if not values:
                print("EMPTY")
                continue

            numbers = [float(v.value) for v in values]

            print("Count:", len(numbers))
            print("Minimum:", min(numbers))
            print("Maximum:", max(numbers))
            print("First:", numbers[0])
            print("Last:", numbers[-1])

            if tag.startswith("test/"):
                print("Checkpoints:")
                for v in values:
                    print(f"  step={v.step}, value={v.value:.8f}")
