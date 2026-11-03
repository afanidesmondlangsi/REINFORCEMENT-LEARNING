import csv
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("experiments_10k_576/results/BC_PAR/bc_par")
OUTPUT = Path("experiments_10k_576/audit_BC_PAR_10k.csv")

ENVS = ["hopper-friction", "halfcheetah-friction",
        "walker2d-friction", "ant-friction"]
SHIFTS = ["0.1", "0.5", "2.0", "5.0"]
SEEDS = range(6)
EXPECTED = set(range(1000, 10001, 1000))

TAGS = ["test/target normalized score",
        "test/target_normalized_score"]

rows = []

for env in ENVS:
    for shift in SHIFTS:
        for seed in SEEDS:
            run = ROOT / f"{env}-srcdatatype-medium-{shift}" / f"r{seed}"
            tb = run / "tb"

            steps = set()
            final_score = None
            event_count = 0
            error = ""

            try:
                files = list(tb.glob("events.out.tfevents.*"))
                event_count = len(files)

                if files:
                    ea = EventAccumulator(
                        str(tb), size_guidance={"scalars": 0}
                    )
                    ea.Reload()

                    tags = ea.Tags().get("scalars", [])
                    tag = next((t for t in TAGS if t in tags), None)

                    if tag:
                        events = ea.Scalars(tag)
                        steps = {e.step for e in events}
                        final = [e for e in events if e.step == 10000]
                        if final:
                            final_score = final[-1].value
                    else:
                        error = "Normalized-score tag missing"
                else:
                    error = "TensorBoard event file missing"

            except Exception as exc:
                error = str(exc)

            missing = sorted(EXPECTED - steps)
            status = "COMPLETE" if not missing and final_score is not None else "INCOMPLETE"

            rows.append({
                "environment": env,
                "shift": shift,
                "seed": seed,
                "status": status,
                "evaluation_count": len(steps),
                "missing_steps": ";".join(map(str, missing)),
                "final_score": final_score,
                "event_files": event_count,
                "error": error
            })

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

complete = sum(r["status"] == "COMPLETE" for r in rows)
incomplete = len(rows) - complete

print("\n========== BC-PAR TENSORBOARD AUDIT ==========")
print(f"Total configurations: {len(rows)}")
print(f"Fully verified:       {complete}")
print(f"Incomplete:           {incomplete}")

print("\nBreakdown by environment:")
for env in ENVS:
    subset = [r for r in rows if r["environment"] == env]
    passed = sum(r["status"] == "COMPLETE" for r in subset)
    print(f"{env:22s} {passed}/24 complete")

print("\nIncomplete runs:")
for r in rows:
    if r["status"] != "COMPLETE":
        print(
            r["environment"], r["shift"],
            f"seed={r['seed']}",
            "missing=", r["missing_steps"],
            "error=", r["error"]
        )

print(f"\nDetailed audit saved to: {OUTPUT}")
