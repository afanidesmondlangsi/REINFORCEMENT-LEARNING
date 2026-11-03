from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()

DATA = ROOT / "matched_final_5000" / "final_three_RQ_analysis" / "all_learning_curve_records.csv"

OUT = ROOT / "matched_final_5000" / "final_learning_curves" / "RQ2_task_relatedness"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

METHODS = [
    "SAC_TARGET_ONLY",
    "BC_SAC",
    "BC_PAR",
    "RLPD",
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED",
]

LABELS = {
    "SAC_TARGET_ONLY": "SAC Target-Only",
    "BC_SAC": "BC-SAC",
    "BC_PAR": "BC-PAR",
    "RLPD": "RLPD",
    "EA_NTC_NORMALIZED": "EA-NTC Normalized",
    "EA_NTC_UNNORMALIZED": "EA-NTC Mean",
}

ENVIRONMENTS = ["ant", "halfcheetah", "hopper", "walker2d"]
MISSING_SHIFTS = [0.1, 0.5]
CHECKPOINTS = [1000, 2000, 3000, 4000, 5000]

print("\n========== RQ2 MISSING FIGURE REPAIR ==========")

generated = 0

for env in ENVIRONMENTS:

    for shift in MISSING_SHIFTS:

        # Explicit labels prevent shift truncation.
        shift_label = f"{shift:.1f}"

        name = f"RQ2_{env}_shift_{shift_label}"

        png_path = OUT / f"{name}.png"
        pdf_path = OUT / f"{name}.pdf"

        # Never overwrite existing figures.
        if png_path.exists() or pdf_path.exists():
            print("SKIPPED EXISTING:", name)
            continue

        subset = df[
            (df["environment"] == env) &
            np.isclose(df["shift"], shift)
        ].copy()

        if subset.empty:
            raise ValueError(
                f"No data found for {env}, shift {shift_label}"
            )

        fig, ax = plt.subplots(figsize=(10, 6.5))

        for method in METHODS:

            method_data = subset[
                subset["method"] == method
            ]

            summary = (
                method_data.groupby("step")["target_return"]
                .agg(["mean", "std", "count"])
                .reindex(CHECKPOINTS)
            )

            if summary["mean"].isna().any():
                raise ValueError(
                    f"Missing checkpoints: {env}, "
                    f"{shift_label}, {method}"
                )

            if not (summary["count"] == 3).all():
                raise ValueError(
                    f"Expected three seeds: {env}, "
                    f"{shift_label}, {method}"
                )

            x = summary.index.to_numpy()
            mean = summary["mean"].to_numpy()
            std = summary["std"].to_numpy()

            highlighted = method in [
                "EA_NTC_NORMALIZED",
                "EA_NTC_UNNORMALIZED",
            ]

            line, = ax.plot(
                x,
                mean,
                marker="o",
                linewidth=2.8 if highlighted else 1.8,
                linestyle="-" if highlighted else "--",
                markersize=6,
                label=LABELS[method],
            )

            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=line.get_color(),
                alpha=0.12 if highlighted else 0.07,
            )

        ax.set_title(
            f"RQ2: Task Relatedness - {env.title()} | Shift {shift_label}",
            fontsize=14,
            fontweight="bold",
        )

        ax.set_xlabel("Environment Interaction Steps", fontsize=12)
        ax.set_ylabel("Mean Target Return", fontsize=12)

        ax.set_xticks(CHECKPOINTS)
        ax.set_xticklabels([
            "1,000", "2,000", "3,000", "4,000", "5,000"
        ])

        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", fontsize=9, ncol=2)

        fig.tight_layout()

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight"
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight"
        )

        plt.close(fig)

        generated += 1

        print(
            f"GENERATED: {env} | shift={shift_label} | "
            f"PNG + PDF"
        )

print("\n========== REPAIR SUMMARY ==========")
print("New figures:", generated)
print("New PNG files:", generated)
print("New PDF files:", generated)

print("\nRQ2 PNG total:", len(list(OUT.glob("*.png"))))
print("RQ2 PDF total:", len(list(OUT.glob("*.pdf"))))

print("\nOutput:", OUT.resolve())

if len(list(OUT.glob("*.png"))) == 16 and len(list(OUT.glob("*.pdf"))) == 16:
    print("\nSUCCESS: All 16 RQ2 figures are present.")
else:
    print("\nWARNING: RQ2 figure count is not yet complete.")
