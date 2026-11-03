import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = r"logs_ea_ntc_norm_vs_mean_4env_6shift_6seed_5000"

for current, dirs, files in os.walk(root):
    for f in files:
        if f.startswith("events.out.tfevents"):
            path = os.path.join(current, f)

            try:
                ea = EventAccumulator(path)
                ea.Reload()

                print("\n" + "=" * 70)
                print("RUN:", current)
                print("=" * 70)

                for tag in ea.Tags().get("scalars", []):
                    events = ea.Scalars(tag)

                    if events:
                        print(
                            f"{tag:45s} "
                            f"count={len(events):5d} "
                            f"first={events[0].step:6d} "
                            f"last={events[-1].step:6d}"
                        )

            except Exception as e:
                print("ERROR:", path, str(e))
