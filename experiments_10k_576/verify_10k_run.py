import argparse
import json
import math
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EXPECTED_STEPS = set(range(1000, 10001, 1000))
SCORE_TAGS = (
    "test/target normalized score",
    "test/target_normalized_score",
)


def verify(run_dir):
    tb_dir = Path(run_dir) / "tb"

    if not tb_dir.is_dir():
        return {"complete": False, "reason": "TensorBoard directory missing"}

    event_files = sorted(tb_dir.glob("events.out.tfevents.*"))

    if not event_files:
        return {"complete": False, "reason": "No TensorBoard event files"}

    scores = {}

    try:
        accumulator = EventAccumulator(
            str(tb_dir),
            size_guidance={"scalars": 0},
        )
        accumulator.Reload()

        available_tags = set(accumulator.Tags().get("scalars", []))
        matching_tags = available_tags.intersection(SCORE_TAGS)

        if not matching_tags:
            return {
                "complete": False,
                "reason": "Target normalized score tag missing",
            }

        for tag in matching_tags:
            for event in accumulator.Scalars(tag):
                if math.isfinite(event.value):
                    scores[int(event.step)] = float(event.value)

    except Exception as exc:
        return {"complete": False, "reason": str(exc)}

    missing = sorted(EXPECTED_STEPS.difference(scores))

    return {
        "complete": not missing,
        "evaluation_count": len(EXPECTED_STEPS.intersection(scores)),
        "missing_steps": missing,
        "final_score": scores.get(10000),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()

    result = verify(args.run_dir)
    print(json.dumps(result, indent=2))

    sys.exit(0 if result["complete"] else 1)
