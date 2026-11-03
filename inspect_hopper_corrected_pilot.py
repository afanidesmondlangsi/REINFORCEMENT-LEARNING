from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path("logs_hopper_corrected_pilot")

for run in sorted(root.iterdir()):

    if not run.is_dir():
        continue

    event_files = list(run.rglob("events.out.tfevents.*"))

    if not event_files:
        print(f"\n{run.name}: NO EVENTS")
        continue

    ea = EventAccumulator(
        str(event_files[0]),
        size_guidance={"scalars": 0}
    )
    ea.Reload()

    print(f"\n========== {run.name} ==========")

    tags = ea.Tags().get("scalars", [])

    for tag in tags:

        if any(
            key in tag.lower()
            for key in [
                "gate",
                "weight",
                "strength",
                "reliab",
                "relative",
                "target_normalized_score"
            ]
        ):

            values = ea.Scalars(tag)

            if values:
                last = values[-1]

                print(
                    f"{tag}: "
                    f"step={last.step}, "
                    f"value={last.value:.6f}"
                )
