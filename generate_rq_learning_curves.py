
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path.cwd()

INPUT = (
    PROJECT
    / "matched_final_5000"
    / "final_three_RQ_analysis"
    / "all_learning_curve_records.csv"
)

OUTPUT = PROJECT / "matched_final_5000" / "final_learning_curves"

RQ1_DIR = OUTPUT / "RQ1_negative_transfer"
RQ2_DIR = OUTPUT / "RQ2_task_relatedness"
RQ3_DIR = OUTPUT / "RQ3_adaptation"

for directory in [RQ1_DIR, RQ2_DIR, RQ3_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

ENVIRONMENTS = ["ant", "halfcheetah", "hopper", "walker2d"]

SHIFTS = [0.1, 0.5, 2.0, 5.0]

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

CHECKPOINTS = [1000, 2000, 3000, 4000, 5000]

EA_METHODS = [
    "EA_NTC_NORMALIZED",
    "EA_NTC_UNNORMALIZED",
]

BASELINES = [
    "BC_SAC",
    "BC_PAR",
    "RLPD",
    "SAC_TARGET_ONLY",
]

# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

print("\n========== LOADING EXPERIMENTAL DATA ==========")

if not INPUT.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT}")

df = pd.read_csv(INPUT)

required = [
    "environment",
    "shift",
    "seed",
    "method",
    "checkpoint",
    "target_return",
    "step",
]

missing = set(required) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["shift"] = pd.to_numeric(df["shift"])
df["seed"] = pd.to_numeric(df["seed"])
df["step"] = pd.to_numeric(df["step"])
df["target_return"] = pd.to_numeric(
    df["target_return"], errors="coerce"
)

df = df.dropna(subset=["target_return"])

df["method"] = df["method"].str.strip()
df["environment"] = df["environment"].str.lower()

# Ensure that every experimental observation is unique.
key = [
    "environment",
    "shift",
    "seed",
    "method",
    "step",
]

duplicates = df.duplicated(key)

if duplicates.any():
    raise ValueError(
        f"Found {duplicates.sum()} duplicate experimental records."
    )

if set(df["method"].unique()) != set(METHODS):
    raise ValueError("Unexpected algorithm coverage.")

if set(df["environment"].unique()) != set(ENVIRONMENTS):
    raise ValueError("Unexpected environment coverage.")

if set(df["step"].unique()) != set(CHECKPOINTS):
    raise ValueError("Unexpected evaluation checkpoints.")

print("Records:", len(df))
print("Environments:", df["environment"].nunique())
print("Algorithms:", df["method"].nunique())
print("Shifts:", df["shift"].nunique())
print("Seeds:", sorted(df["seed"].unique()))

# ============================================================
# COMMON PLOTTING FUNCTION
# ============================================================

def plot_learning_curves(
    data,
    methods,
    title,
    subtitle,
    output_path,
    emphasis=None,
    shared_ylim=None,
):
    """
    Plot mean target return with +/- one standard deviation.
    """

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for method in methods:

        subset = data[data["method"] == method]

        summary = (
            subset.groupby("step")["target_return"]
            .agg(["mean", "std", "count"])
            .reindex(CHECKPOINTS)
        )

        if summary["mean"].isna().any():
            raise ValueError(
                f"Missing checkpoint for {method}: {title}"
            )

        x = summary.index.to_numpy()
        mean = summary["mean"].to_numpy()
        std = summary["std"].fillna(0).to_numpy()

        highlighted = method in (emphasis or [])

        line, = ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2.8 if highlighted else 1.8,
            markersize=6,
            linestyle="-" if highlighted else "--",
            label=LABELS[method],
        )

        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.12 if highlighted else 0.07,
        )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)

    ax.text(
        0.5,
        1.01,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )

    ax.set_xlabel("Environment Interaction Steps", fontsize=12)
    ax.set_ylabel("Mean Target Return", fontsize=12)

    ax.set_xticks(CHECKPOINTS)
    ax.set_xticklabels(
        ["1,000", "2,000", "3,000", "4,000", "5,000"]
    )

    if shared_ylim is not None:
        ax.set_ylim(shared_ylim)

    ax.grid(True, linestyle=":", alpha=0.5)

    ax.legend(
        loc="best",
        fontsize=9,
        frameon=True,
        ncol=2,
    )

    fig.tight_layout()

    # Publication-quality PNG and vector PDF.
    fig.savefig(
        output_path.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# RQ1: NEGATIVE TRANSFER
# ============================================================

print("\n========== GENERATING RQ1 ==========")

for env in ENVIRONMENTS:

    subset = df[df["environment"] == env].copy()

    plot_learning_curves(
        data=subset,
        methods=METHODS,
        title=f"RQ1: Negative Transfer - {env.title()}",
        subtitle=(
            "Six-algorithm comparison | "
            "Four friction shifts | Three seeds"
        ),
        output_path=RQ1_DIR / f"RQ1_learning_curve_{env}",
        emphasis=EA_METHODS,
    )

    print(f"Generated RQ1: {env}")


# ============================================================
# RQ2: TASK RELATEDNESS
# ============================================================

print("\n========== GENERATING RQ2 ==========")

for env in ENVIRONMENTS:

    env_data = df[df["environment"] == env]

    # Use common y-axis limits across all four shifts
    # within the same environment.
    low = env_data["target_return"].min()
    high = env_data["target_return"].max()

    padding = max((high - low) * 0.08, 1)

    shared_ylim = (
        low - padding,
        high + padding,
    )

    for shift in SHIFTS:

        subset = env_data[
            np.isclose(env_data["shift"], shift)
        ].copy()

        plot_learning_curves(
            data=subset,
            methods=METHODS,
            title=(
                f"RQ2: Task Relatedness - "
                f"{env.title()} | Shift {shift}"
            ),
            subtitle=(
                f"Friction shift = {shift} | "
                "Three matched seeds | Six algorithms"
            ),
            output_path=(
                RQ2_DIR
                / f"RQ2_{env}_shift_{shift}"
            ),
            emphasis=EA_METHODS,
            shared_ylim=shared_ylim,
        )

        print(f"Generated RQ2: {env}, shift={shift}")


# ============================================================
# RQ3: EA-NTC ADAPTATION
# ============================================================

print("\n========== GENERATING RQ3 ==========")

for env in ENVIRONMENTS:

    subset = df[df["environment"] == env].copy()

    # Keep all four established baselines visible.
    # Highlight both EA-NTC formulations.
    plot_learning_curves(
        data=subset,
        methods=[
            "SAC_TARGET_ONLY",
            "BC_SAC",
            "BC_PAR",
            "RLPD",
            "EA_NTC_NORMALIZED",
            "EA_NTC_UNNORMALIZED",
        ],
        title=f"RQ3: EA-NTC Adaptation - {env.title()}",
        subtitle=(
            "EA-NTC Normalized vs EA-NTC Mean "
            "against four established baselines | "
            "Four shifts | Three seeds"
        ),
        output_path=RQ3_DIR / f"RQ3_adaptation_{env}",
        emphasis=EA_METHODS,
    )

    print(f"Generated RQ3: {env}")


# ============================================================
# FINAL OUTPUT SUMMARY
# ============================================================

png_files = list(OUTPUT.rglob("*.png"))
pdf_files = list(OUTPUT.rglob("*.pdf"))

print("\n" + "=" * 60)
print("LEARNING CURVE GENERATION COMPLETED")
print("=" * 60)

print("RQ1 figures:", len(list(RQ1_DIR.glob("*.png"))))
print("RQ2 figures:", len(list(RQ2_DIR.glob("*.png"))))
print("RQ3 figures:", len(list(RQ3_DIR.glob("*.png"))))

print("Total PNG figures:", len(png_files))
print("Total PDF figures:", len(pdf_files))

print("\nOutput directory:")
print(OUTPUT.resolve())

print("\nAll figures include:")
print("- Mean target return")
print("- Standard deviation bands")
print("- Five evaluation checkpoints")
print("- Consistent algorithm labels")
print("- 300 DPI PNG and vector PDF")

print("\nSUCCESS")
