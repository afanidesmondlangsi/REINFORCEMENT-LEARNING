import sys
import math
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir = Path(sys.argv[1])

required_tags = [
    "test/target_return",
    "test/target_normalized_score"
]

required_steps = {1000, 2000, 3000, 4000, 5000}

event_files = list(run_dir.rglob("events.out.tfevents.*"))

if not event_files:
    print("PENDING: No TensorBoard event files found.")
    sys.exit(1)

steps_by_tag = {tag: set() for tag in required_tags}

for event_dir in set(f.parent for f in event_files):
    accumulator = EventAccumulator(
        str(event_dir),
        size_guidance={"scalars": 0}
    )
    accumulator.Reload()

    available_tags = accumulator.Tags().get("scalars", [])

    for tag in required_tags:
        if tag not in available_tags:
            continue

        for event in accumulator.Scalars(tag):
            if math.isfinite(event.value):
                steps_by_tag[tag].add(event.step)

for tag in required_tags:
    missing = sorted(required_steps - steps_by_tag[tag])

    if missing:
        print(f"PENDING: {tag}, missing steps: {missing}")
        sys.exit(1)

print("COMPLETE: All five evaluations verified.")
sys.exit(0)
