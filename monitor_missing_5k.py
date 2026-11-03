import csv
import os
import time
from pathlib import Path
from datetime import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

MANIFEST = Path("experiments_10k_576/experiment_manifest_missing_5k.csv")
INTERVAL = 10

def get_status(run_dir):
    if not run_dir.exists():
        return "PENDING"

    files = list(run_dir.rglob("events.out.tfevents.*"))
    if not files:
        return "STARTED"

    max_steps = {
        "test/target_return": 0,
        "test/target_normalized_score": 0
    }

    for event_file in files:
        try:
            ea = EventAccumulator(
                str(event_file.parent),
                size_guidance={"scalars": 0}
            )
            ea.Reload()
            tags = ea.Tags().get("scalars", [])

            for tag in max_steps:
                if tag in tags:
                    values = ea.Scalars(tag)
                    if values:
                        max_steps[tag] = max(
                            max_steps[tag],
                            max(v.step for v in values)
                        )
        except Exception:
            continue

    if all(step >= 5000 for step in max_steps.values()):
        return "COMPLETE"

    return "STARTED"

def main():
    if not MANIFEST.exists():
        print("Manifest not found:", MANIFEST)
        return

    with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        runs = list(csv.DictReader(f))

    total = len(runs)

    while True:
        results = []

        for run in runs:
            run_dir = Path(run["RunDirectory"])
            status = get_status(run_dir)

            results.append({
                "algorithm": run["Algorithm"],
                "environment": run["Environment"],
                "shift": run["Shift"],
                "seed": run["Seed"],
                "status": status
            })

        completed = sum(r["status"] == "COMPLETE" for r in results)
        started = sum(r["status"] == "STARTED" for r in results)
        pending = sum(r["status"] == "PENDING" for r in results)
        remaining = total - completed
        percentage = completed / total * 100 if total else 0

        os.system("cls" if os.name == "nt" else "clear")

        print("=" * 65)
        print("             LIVE 5,000-STEP EXPERIMENT MONITOR")
        print("=" * 65)
        print("Last updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print()
        print(f"TOTAL EXPERIMENTS : {total}")
        print(f"COMPLETED         : {completed}")
        print(f"STARTED           : {started}")
        print(f"PENDING           : {pending}")
        print(f"REMAINING         : {remaining}")
        print()

        bar_width = 40
        filled = int(bar_width * percentage / 100)
        bar = "#" * filled + "-" * (bar_width - filled)

        print(f"PROGRESS: [{bar}] {percentage:.1f}%")
        print()
        print("-" * 65)

        for r in results:
            name = (
                f'{r["algorithm"]}_{r["environment"]}_'
                f'{r["shift"]}_seed{r["seed"]}'
            )
            print(f'{r["status"]:10s}: {name}')

        print("-" * 65)
        print(f"Next refresh in {INTERVAL} seconds.")
        print("Press CTRL+C to stop monitoring.")

        time.sleep(INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
