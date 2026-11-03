import csv
import hashlib
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(r"C:\Users\LANGS\Documents\off-dynamics-rl")
OUT = ROOT / "ea_ntc_phase17_phase19_tensorboard_audit"
OUT.mkdir(exist_ok=True)

EXPERIMENTS = [
    "logs_ea_ntc_phase17a_adaptive",
    "logs_ea_ntc_phase17b_normalized",
    "logs_ea_ntc_phase17c_confirmation",
    "logs_phase19",
    "logs_phase19_10k",
    "logs_phase19_10k_kappa05",
]

STEPS = [1000, 2000, 3000, 4000, 5000, 10000]

TAGS = {
    "raw": ["test/target_return", "test/target return"],
    "normalized": [
        "test/target_normalized_score",
        "test/target normalized score"
    ]
}

def read_tag(acc, aliases):
    available = set(acc.Tags().get("scalars", []))
    tag = next((x for x in aliases if x in available), None)
    if tag is None:
        return None, {}
    values = {
        int(e.step): float(e.value)
        for e in acc.Scalars(tag)
    }
    return tag, values

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

records = []
long_records = []
errors = []

for experiment in EXPERIMENTS:
    exp_root = ROOT / experiment

    if not exp_root.exists():
        errors.append({
            "experiment": experiment,
            "reason": "EXPERIMENT_DIRECTORY_MISSING"
        })
        continue

    for tb in exp_root.rglob("tb"):
        if not tb.is_dir():
            continue

        events = sorted(tb.glob("events.out.tfevents.*"))

        if not events:
            continue

        try:
            acc = EventAccumulator(
                str(tb),
                size_guidance={"scalars": 0}
            )
            acc.Reload()

            raw_tag, raw = read_tag(acc, TAGS["raw"])
            norm_tag, norm = read_tag(acc, TAGS["normalized"])

            all_steps = sorted(set(raw) | set(norm))

            event_hashes = [
                sha256(f) for f in events
            ]

            row = {
                "experiment": experiment,
                "run_path": str(tb.relative_to(ROOT)),
                "raw_tag": raw_tag or "",
                "normalized_tag": norm_tag or "",
                "raw_checkpoint_count": len(raw),
                "normalized_checkpoint_count": len(norm),
                "last_raw_step": max(raw) if raw else "",
                "last_normalized_step": max(norm) if norm else "",
                "raw_complete_5000": all(s in raw for s in STEPS[:5]),
                "normalized_complete_5000": all(s in norm for s in STEPS[:5]),
                "raw_complete_10000": all(s in raw for s in STEPS),
                "normalized_complete_10000": all(s in norm for s in STEPS),
                "event_file_count": len(events),
                "event_hashes": ";".join(event_hashes)
            }

            for step in STEPS:
                row[f"raw_{step}"] = raw.get(step, "")
                row[f"normalized_{step}"] = norm.get(step, "")

            records.append(row)

            for step in all_steps:
                long_records.append({
                    "experiment": experiment,
                    "run_path": str(tb.relative_to(ROOT)),
                    "step": step,
                    "raw_return": raw.get(step, ""),
                    "normalized_score": norm.get(step, "")
                })

        except Exception as exc:
            errors.append({
                "experiment": experiment,
                "reason": repr(exc),
                "path": str(tb)
            })

def write_csv(filename, rows):
    if not rows:
        print("No records:", filename)
        return

    keys = list(dict.fromkeys(
        key for row in rows for key in row
    ))

    with (OUT / filename).open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

write_csv("evaluation_checkpoints.csv", records)
write_csv("evaluation_long_format.csv", long_records)
write_csv("audit_errors.csv", errors)

print("\nPHASE 17-19 TENSORBOARD AUDIT")
print("=" * 55)
print("Total runs:", len(records))
print("Errors:", len(errors))

for experiment in EXPERIMENTS:
    subset = [r for r in records if r["experiment"] == experiment]

    print(f"\n{experiment}")
    print("Runs:", len(subset))
    print("Complete raw 5000:",
          sum(r["raw_complete_5000"] for r in subset))
    print("Complete normalized 5000:",
          sum(r["normalized_complete_5000"] for r in subset))
    print("Complete raw 10000:",
          sum(r["raw_complete_10000"] for r in subset))

print("\nReports saved to:", OUT)
print("READ-ONLY: no training or historical log modification.")
