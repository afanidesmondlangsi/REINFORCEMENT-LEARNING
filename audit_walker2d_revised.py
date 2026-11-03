from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import csv
import numpy as np

root = Path("logs_walker2d_revised_gate_48runs")
shifts = ("0.1", "0.5", "2.0", "5.0")
seeds = range(6)
treatments = ("NORMALIZED", "UNNORMALIZED")

rows = []
curves = []
expected_steps = [1000, 2000, 3000, 4000, 5000]

def scalar_values(ea, tag):
    if tag not in ea.Tags().get("scalars", []):
        return []
    return ea.Scalars(tag)

for treatment in treatments:
    for shift in shifts:
        for seed in seeds:

            name = f"{treatment}_walker2d_shift_{shift}_seed_{seed}"
            directory = root / name
            files = list(directory.rglob("events.out.tfevents.*"))

            row = {
                "treatment": treatment,
                "shift": shift,
                "seed": seed,
                "status": "MISSING_EVENTS",
                "gate_ever_ready": False,
                "max_gate_strength": np.nan,
                "min_source_weight": np.nan,
                "final_score": np.nan,
                "auc": np.nan,
            }

            if len(files) != 1:
                row["status"] = f"EVENT_FILE_COUNT_{len(files)}"
                rows.append(row)
                continue

            try:
                ea = EventAccumulator(
                    str(files[0].parent),
                    size_guidance={"scalars": 0},
                )
                ea.Reload()

                gate = scalar_values(ea, "ea_ntc/gate_ready")
                strength = scalar_values(
                    ea, "ea_ntc/adaptive_gate_strength"
                )
                weights = scalar_values(
                    ea, "ea_ntc/mean_source_weight"
                )
                scores = scalar_values(
                    ea, "test/target_normalized_score"
                )

                if not all((gate, strength, weights, scores)):
                    row["status"] = "MISSING_SCALARS"
                    rows.append(row)
                    continue

                row["gate_ever_ready"] = any(
                    v.value > 0.5 for v in gate
                )
                row["max_gate_strength"] = max(
                    v.value for v in strength
                )
                row["min_source_weight"] = min(
                    v.value for v in weights
                )

                score_map = {v.step: v.value for v in scores}

                if sorted(score_map) != expected_steps:
                    row["status"] = "INCOMPLETE_CHECKPOINTS"
                    rows.append(row)
                    continue

                values = [
                    score_map[step] for step in expected_steps
                ]

                row["final_score"] = values[-1]
                row["auc"] = float(
                    np.trapz(values, x=expected_steps)
                )
                row["status"] = "OK"

                for step, value in zip(expected_steps, values):
                    curves.append({
                        "treatment": treatment,
                        "shift": shift,
                        "seed": seed,
                        "step": step,
                        "score": value,
                    })

            except Exception as exc:
                row["status"] = f"ERROR: {exc}"

            rows.append(row)

def save_csv(filename, data):
    if not data:
        return
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

save_csv("walker2d_revised_audit.csv", rows)
save_csv("walker2d_revised_curves.csv", curves)

print("\n========== RUN VALIDATION ==========")
print("Total:", len(rows))
print("Valid:", sum(r["status"] == "OK" for r in rows))
print("Gate activated:", sum(r["gate_ever_ready"] for r in rows))
print("Positive strength:", sum(
    np.isfinite(r["max_gate_strength"])
    and r["max_gate_strength"] > 0
    for r in rows
))
print("Nonunit weights:", sum(
    np.isfinite(r["min_source_weight"])
    and r["min_source_weight"] < 0.999999
    for r in rows
))

print("\n========== PAIRED COMPARISON ==========")

pairs = []

for shift in shifts:
    for seed in seeds:
        matched = {
            r["treatment"]: r
            for r in rows
            if r["shift"] == shift
            and r["seed"] == seed
            and r["status"] == "OK"
        }

        if len(matched) != 2:
            continue

        n = matched["NORMALIZED"]
        u = matched["UNNORMALIZED"]

        pairs.append({
            "shift": shift,
            "seed": seed,
            "auc_difference": n["auc"] - u["auc"],
            "final_score_difference":
                n["final_score"] - u["final_score"],
        })

save_csv("walker2d_revised_paired.csv", pairs)

for shift in shifts:
    group = [p for p in pairs if p["shift"] == shift]

    if not group:
        print(shift, "NO VALID PAIRS")
        continue

    auc = np.array([p["auc_difference"] for p in group])
    final = np.array([
        p["final_score_difference"] for p in group
    ])

    print(
        f"Shift {shift}: "
        f"pairs={len(group)}, "
        f"mean AUC difference={auc.mean():.3f}, "
        f"mean final difference={final.mean():.3f}, "
        f"normalized wins={sum(auc > 0)}/{len(auc)}"
    )

print("\nSaved audit, curves, and paired comparison CSV files.")
