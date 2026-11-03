import argparse
import time
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--interval", type=int, default=5)
args = parser.parse_args()

tb = Path(args.run_dir) / "tb"
last_step = -1

print("Monitoring training. Press Ctrl+C to stop monitoring only.", flush=True)

try:
    while True:
        if tb.exists():
            try:
                ea = EventAccumulator(str(tb), size_guidance={"scalars": 0})
                ea.Reload()

                tags = ea.Tags().get("scalars", [])
                tag = next(
                    (t for t in (
                        "test/target normalized score",
                        "test/target_normalized_score"
                    ) if t in tags),
                    None
                )

                if tag:
                    events = ea.Scalars(tag)

                    if events:
                        latest = max(events, key=lambda e: e.step)

                        if latest.step != last_step:
                            last_step = latest.step
                            progress = 100 * latest.step / 10000

                            print(
                                f"Step {latest.step:5d}/10000 | "
                                f"Progress {progress:5.1f}% | "
                                f"Normalized score {latest.value:.4f}",
                                flush=True
                            )

                            if latest.step >= 10000:
                                print("All ten evaluations recorded.", flush=True)
                                break

            except Exception as exc:
                print(f"Waiting for readable TensorBoard data: {exc}", flush=True)

        time.sleep(args.interval)

except KeyboardInterrupt:
    print("\nMonitoring stopped. Training was not interrupted.")
