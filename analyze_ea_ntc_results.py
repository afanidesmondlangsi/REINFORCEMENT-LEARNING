import csv
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

ROOT = Path(r"C:\Users\LANGS\Documents\off-dynamics-rl")
INPUT = ROOT / "ea_ntc_phase17_phase19_tensorboard_audit" / "evaluation_checkpoints.csv"
OUT = ROOT / "ea_ntc_quantitative_results"
OUT.mkdir(exist_ok=True)

STEPS_5K = [1000, 2000, 3000, 4000, 5000]
STEPS_10K = list(range(1000, 10001, 1000))

def number(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None

def treatment(row):
    exp = row["experiment"]
    path = row["run_path"].lower()

    if "phase17a" in exp:
        return "PHASE17A_ADAPTIVE_GATE"

    if "phase17b" in exp:
        return "PHASE17B_NORMALIZED"

    if "phase17c" in exp:
        if "mean_shift" in path:
            return "PHASE17C_MEAN"
        if "normalized_shift" in path:
            return "PHASE17C_NORMALIZED"

    if "phase19_10k_kappa05" in exp:
        return "PHASE19_KAPPA_0.5"

    if "phase19_10k" in exp:
        return "PHASE19_10K"

    if exp == "logs_phase19":
        return "PHASE19_STANDARD"

    return "UNCLASSIFIED"

def get_values(row, metric, steps):
    values = []
    for step in steps:
        value = number(row.get(f"{metric}_{step}"))
        if value is None:
            return None
        values.append(value)
    return values

def auc(values, steps):
    if values is None:
        return None

    area = sum(
        (steps[i+1] - steps[i]) *
        (values[i] + values[i+1]) / 2
        for i in range(len(steps)-1)
    )

    return area

def stats(values):
    if not values:
        return None, None, 0
    return mean(values), stdev(values) if len(values) > 1 else 0.0, len(values)

with INPUT.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    row["treatment"] = treatment(row)
    row["shift"] = next(
        (s for s in ["0.1", "0.5", "2.0", "5.0"]
         if f"medium-{s}" in row["run_path"]),
        "unknown"
    )

    # Extract seed from run directory.
    import re
    match = re.search(r"(?:^|[\\/])r(\d+)(?:[\\/]|$)", row["run_path"])
    row["seed"] = int(match.group(1)) if match else None

print("Total records:", len(rows))

# --------------------------------------------------
# A. CHECKPOINT SUMMARY
# --------------------------------------------------

checkpoint_summary = []

for treatment_name in sorted(set(r["treatment"] for r in rows)):
    for shift in sorted(set(r["shift"] for r in rows)):
        subset = [
            r for r in rows
            if r["treatment"] == treatment_name
            and r["shift"] == shift
        ]

        for metric in ["raw", "normalized"]:
            for step in STEPS_5K:
                values = [
                    number(r.get(f"{metric}_{step}"))
                    for r in subset
                ]
                values = [v for v in values if v is not None]

                m, sd, n = stats(values)

                checkpoint_summary.append({
                    "treatment": treatment_name,
                    "shift": shift,
                    "metric": metric,
                    "step": step,
                    "n": n,
                    "mean": m,
                    "sd": sd
                })

# --------------------------------------------------
# B. AUC SUMMARY
# --------------------------------------------------

auc_records = []

for row in rows:
    treatment_name = row["treatment"]

    for metric in ["raw", "normalized"]:
        values5 = get_values(row, metric, STEPS_5K)

        auc5 = auc(values5, STEPS_5K)

        values10 = get_values(row, metric, STEPS_10K)
        auc10 = auc(values10, STEPS_10K)

        auc_records.append({
            "treatment": treatment_name,
            "experiment": row["experiment"],
            "shift": row["shift"],
            "seed": row["seed"],
            "run_path": row["run_path"],
            "metric": metric,
            "auc_1k_5k": auc5,
            "normalized_auc_1k_5k":
                auc5 / 4000 if auc5 is not None else None,
            "auc_1k_10k": auc10,
            "normalized_auc_1k_10k":
                auc10 / 9000 if auc10 is not None else None
        })

# --------------------------------------------------
# C. AUC GROUP SUMMARY
# --------------------------------------------------

auc_summary = []

for treatment_name in sorted(set(r["treatment"] for r in rows)):
    for shift in sorted(set(r["shift"] for r in rows)):
        for metric in ["raw", "normalized"]:
            subset = [
                r for r in auc_records
                if r["treatment"] == treatment_name
                and r["shift"] == shift
                and r["metric"] == metric
                and r["auc_1k_5k"] is not None
            ]

            values = [r["auc_1k_5k"] for r in subset]
            m, sd, n = stats(values)

            auc_summary.append({
                "treatment": treatment_name,
                "shift": shift,
                "metric": metric,
                "n": n,
                "mean_auc_1k_5k": m,
                "sd_auc_1k_5k": sd,
                "mean_auc_per_step":
                    m / 4000 if m is not None else None
            })

# --------------------------------------------------
# D. PAIRED MEAN VS NORMALIZED BC
# --------------------------------------------------

paired = []

for shift in ["0.1", "0.5"]:
    for seed in range(6):

        mean_run = next((
            r for r in rows
            if r["treatment"] == "PHASE17C_MEAN"
            and r["shift"] == shift
            and r["seed"] == seed
        ), None)

        norm_run = next((
            r for r in rows
            if r["treatment"] in [
                "PHASE17B_NORMALIZED",
                "PHASE17C_NORMALIZED"
            ]
            and r["shift"] == shift
            and r["seed"] == seed
        ), None)

        if mean_run is None or norm_run is None:
            continue

        for metric in ["raw", "normalized"]:
            a = auc(
                get_values(mean_run, metric, STEPS_5K),
                STEPS_5K
            )
            b = auc(
                get_values(norm_run, metric, STEPS_5K),
                STEPS_5K
            )

            if a is not None and b is not None:
                paired.append({
                    "shift": shift,
                    "seed": seed,
                    "metric": metric,
                    "mean_bc_auc": a,
                    "normalized_bc_auc": b,
                    "normalized_minus_mean": b-a
                })

# --------------------------------------------------
# E. PHASE 19 SENSITIVITY COMPARISON
# --------------------------------------------------

phase19 = []

standard = next(
    (r for r in rows if r["treatment"] == "PHASE19_STANDARD"
     and r["raw_complete_5000"] == "True"),
    None
)

extended = next(
    (r for r in rows if r["treatment"] == "PHASE19_10K"),
    None
)

kappa = next(
    (r for r in rows if r["treatment"] == "PHASE19_KAPPA_0.5"),
    None
)

comparisons = [
    ("STANDARD_VS_KAPPA05", standard, kappa),
    ("STANDARD_VS_10K", standard, extended),
    ("10K_VS_KAPPA05", extended, kappa)
]

for label, first, second in comparisons:
    if first is None or second is None:
        continue

    for metric in ["raw", "normalized"]:
        for step in STEPS_5K:
            a = number(first.get(f"{metric}_{step}"))
            b = number(second.get(f"{metric}_{step}"))

            if a is not None and b is not None:
                phase19.append({
                    "comparison": label,
                    "metric": metric,
                    "step": step,
                    "first_value": a,
                    "second_value": b,
                    "second_minus_first": b-a
                })

# --------------------------------------------------
# F. WRITE REPORTS
# --------------------------------------------------

def save_csv(filename, records):
    if not records:
        print("No records:", filename)
        return

    with (OUT / filename).open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

save_csv("checkpoint_summary.csv", checkpoint_summary)
save_csv("individual_auc.csv", auc_records)
save_csv("auc_group_summary.csv", auc_summary)
save_csv("paired_mean_vs_normalized.csv", paired)
save_csv("phase19_sensitivity.csv", phase19)

# --------------------------------------------------
# G. PRINT MAIN RESULTS
# --------------------------------------------------

print("\nCHECKPOINT SUMMARY")
print("=" * 75)

for r in checkpoint_summary:
    if r["metric"] == "raw" and r["step"] == 5000:
        print(
            f'{r["treatment"]:28s} '
            f'shift={r["shift"]:5s} '
            f'n={r["n"]} '
            f'mean={r["mean"]} '
            f'SD={r["sd"]}'
        )

print("\nAUC SUMMARY")
print("=" * 75)

for r in auc_summary:
    if r["metric"] == "raw":
        print(
            f'{r["treatment"]:28s} '
            f'shift={r["shift"]:5s} '
            f'n={r["n"]} '
            f'AUC={r["mean_auc_1k_5k"]} '
            f'SD={r["sd_auc_1k_5k"]}'
        )

print("\nPAIRED MEAN VS NORMALIZED BC")
print("=" * 75)

for r in paired:
    print(r)

print("\nPHASE 19 SENSITIVITY")
print("=" * 75)

for r in phase19:
    print(r)

print("\nALL REPORTS SAVED TO:")
print(OUT)
