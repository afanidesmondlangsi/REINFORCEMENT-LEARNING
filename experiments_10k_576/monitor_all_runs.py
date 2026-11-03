import os
import time
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(r".\experiments_10k_576\results\BC_PAR\bc_par")
ENVIRONMENTS = [
    "hopper-friction",
    "halfcheetah-friction",
    "walker2d-friction",
    "ant-friction",
]
SHIFTS = ["0.1", "0.5", "2.0", "5.0"]
SEEDS = range(6)
MAX_STEP = 10000
INTERVAL = 10

TAG_OPTIONS = [
    "test/target normalized score",
    "test/target_normalized_score",
]

def inspect_run(run_dir):
    tb = run_dir / "tb"

    if not tb.exists():
        return "NOT STARTED", 0, 0, None

    try:
        ea = EventAccumulator(str(tb), size_guidance={"scalars": 0})
        ea.Reload()

        tags = ea.Tags().get("scalars", [])
        tag = next((t for t in TAG_OPTIONS if t in tags), None)

        if tag is None:
            return "INITIALIZING", 0, 0, None

        events = ea.Scalars(tag)

        if not events:
            return "INITIALIZING", 0, 0, None

        latest = max(events, key=lambda e: e.step)
        count = len(events)

        if latest.step >= MAX_STEP and count >= 10:
            status = "COMPLETE"
        else:
            status = "HAS DATA"

        return status, latest.step, count, latest.value

    except Exception:
        return "READING", 0, 0, None


try:
    while True:
        rows = []
        counts = {
            "COMPLETE": 0,
            "HAS DATA": 0,
            "NOT STARTED": 0,
            "INITIALIZING": 0,
            "READING": 0,
        }

        for env in ENVIRONMENTS:
            for shift in SHIFTS:
                folder = f"{env}-srcdatatype-medium-{shift}"

                for seed in SEEDS:
                    run_dir = ROOT / folder / f"r{seed}"
                    status, step, evaluations, score = inspect_run(run_dir)

                    counts[status] += 1

                    progress = 100 * step / MAX_STEP
                    score_text = f"{score:.4f}" if score is not None else "-"

                    rows.append((
                        env, shift, seed, status,
                        step, progress, evaluations, score_text
                    ))

        os.system("cls" if os.name == "nt" else "clear")

        print("=" * 112)
        print("                 BC-PAR LIVE EXPERIMENT DASHBOARD")
        print("=" * 112)
        print(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total configurations: {len(rows)}")
        print()
        print(
            f"COMPLETE: {counts['COMPLETE']:3d} | "
            f"HAS DATA: {counts['HAS DATA']:3d} | "
            f"NOT STARTED: {counts['NOT STARTED']:3d} | "
            f"INITIALIZING: {counts['INITIALIZING']:3d}"
        )
        print("-" * 112)

        print(
            f"{'ENVIRONMENT':<19} {'SHIFT':<7} {'SEED':<6} "
            f"{'STATUS':<13} {'STEP':>7} {'PROGRESS':>9} "
            f"{'EVALS':>7} {'SCORE':>11}"
        )
        print("-" * 112)

        for row in rows:
            env, shift, seed, status, step, progress, evaluations, score = row

            print(
                f"{env:<19} {shift:<7} {seed:<6} "
                f"{status:<13} {step:>7} {progress:>8.1f}% "
                f"{evaluations:>7} {score:>11}"
            )

        print("-" * 112)
        print(
            f"VERIFIED COMPLETE: {counts['COMPLETE']}/96 | "
            f"REMAINING: {96-counts['COMPLETE']}/96"
        )
        print("Refreshing every 10 seconds. Press Ctrl+C to stop monitoring only.")

        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\nDashboard stopped. Training was not interrupted.")
