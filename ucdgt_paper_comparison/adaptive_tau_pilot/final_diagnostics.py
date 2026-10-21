from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("ucdgt_paper_comparison/adaptive_tau_pilot/analysis")
OUT = ROOT / "final_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_ESS = 0.85
INITIAL_TAU = 10.0

runs = pd.read_csv(ROOT / "all_adaptive_run_metrics.csv")
curves = pd.read_csv(ROOT / "all_adaptive_learning_curves.csv")
traj = pd.read_csv(ROOT / "all_adaptive_controller_trajectories.csv")

KEYS = ["environment", "shift", "seed"]

required_metrics = [
    "temperature", "ess_fraction", "update_count"
]

for metric in required_metrics:
    if metric not in set(traj["metric"]):
        raise RuntimeError(f"Missing controller metric: {metric}")

if len(runs) != 24:
    raise RuntimeError(f"Expected 24 runs; found {len(runs)}")

if runs.duplicated(KEYS).any():
    raise RuntimeError("Duplicate adaptive run identifiers")

# ============================================================
# 1. VALIDATE AND ALIGN CONTROLLER UPDATE TRAJECTORIES
# ============================================================

temperature = traj[
    traj["metric"] == "temperature"
].copy()

ess = traj[
    traj["metric"] == "ess_fraction"
].copy()

updates = traj[
    traj["metric"] == "update_count"
].copy()

def prepare_metric(df, value_name):
    df = df[KEYS + ["training_iteration", "value"]].copy()
    df = df.sort_values(KEYS + ["training_iteration"])
    df["update_index"] = df.groupby(KEYS).cumcount() + 1
    return df.rename(columns={
        "value": value_name,
        "training_iteration": f"{value_name}_iteration"
    })

temperature = prepare_metric(temperature, "tau")
ess = prepare_metric(ess, "ess")
updates = prepare_metric(updates, "logged_update_count")

controller = temperature.merge(
    ess, on=KEYS + ["update_index"], validate="one_to_one"
).merge(
    updates, on=KEYS + ["update_index"], validate="one_to_one"
)

counts = controller.groupby(KEYS).size()

if len(counts) != 24 or not (counts == 8).all():
    raise RuntimeError(
        "Expected eight aligned controller updates for each "
        "of the 24 runs. Inspect the TensorBoard logs."
    )

# Require matching logging iterations, rather than assuming
# that independent records always refer to the same update.
if not (
    (controller["tau_iteration"] == controller["ess_iteration"]).all()
    and (
        controller["tau_iteration"]
        == controller["logged_update_count_iteration"]
    ).all()
):
    raise RuntimeError(
        "Temperature, ESS, and update-count timestamps "
        "are not aligned."
    )

if not np.allclose(
    controller["logged_update_count"],
    controller["update_index"]
):
    raise RuntimeError("Logged update counts do not match order")

controller["ess_error"] = controller["ess"] - TARGET_ESS
controller["absolute_ess_error"] = controller["ess_error"].abs()
controller["tau_change"] = controller["tau"] - INITIAL_TAU

controller.to_csv(
    OUT / "controller_aligned_updates.csv", index=False
)

# ============================================================
# 2. RUN-LEVEL CONTROLLER DIAGNOSTICS
# ============================================================

records = []

for key, group in controller.groupby(KEYS):
    env, shift, seed = key
    group = group.sort_values("update_index")

    tau = group["tau"].to_numpy()
    ess_values = group["ess"].to_numpy()
    errors = ess_values - TARGET_ESS

    # Variation between consecutive recorded updates
    tau_steps = np.diff(tau)

    records.append({
        "environment": env,
        "shift": shift,
        "seed": seed,
        "initial_tau_config": INITIAL_TAU,
        "first_logged_tau": tau[0],
        "final_tau": tau[-1],
        "tau_net_change": tau[-1] - INITIAL_TAU,
        "tau_recorded_range": np.ptp(tau),
        "tau_step_total_abs": np.abs(tau_steps).sum(),
        "tau_increase_steps": int((tau_steps > 1e-8).sum()),
        "tau_decrease_steps": int((tau_steps < -1e-8).sum()),
        "mean_ess": ess_values.mean(),
        "final_ess": ess_values[-1],
        "ess_mae": np.abs(errors).mean(),
        "ess_bias": errors.mean(),
        "ess_above_target_fraction": np.mean(errors > 0),
        "ess_below_target_fraction": np.mean(errors < 0),
        "ess_first": ess_values[0],
        "ess_last": ess_values[-1],
        "ess_absolute_error_change": (
            abs(errors[-1]) - abs(errors[0])
        ),
        "controller_updates": len(group)
    })

diagnostics = pd.DataFrame(records)

diagnostics = diagnostics.merge(
    runs[KEYS + ["auc", "final_score", "peak_score"]],
    on=KEYS,
    validate="one_to_one"
)

diagnostics["peak_to_final_drop"] = (
    diagnostics["peak_score"] - diagnostics["final_score"]
)

diagnostics["peak_to_final_drop_fraction"] = np.where(
    diagnostics["peak_score"].abs() > 1e-12,
    diagnostics["peak_to_final_drop"]
    / diagnostics["peak_score"].abs(),
    np.nan
)

diagnostics.to_csv(
    OUT / "controller_run_diagnostics.csv", index=False
)

# ============================================================
# 3. CONDITION-LEVEL SUMMARY
# ============================================================

condition = diagnostics.groupby(
    ["environment", "shift"]
).agg(
    mean_auc=("auc", "mean"),
    sd_auc=("auc", "std"),
    mean_final_score=("final_score", "mean"),
    mean_final_tau=("final_tau", "mean"),
    mean_ess=("mean_ess", "mean"),
    mean_final_ess=("final_ess", "mean"),
    mean_ess_mae=("ess_mae", "mean"),
    mean_ess_bias=("ess_bias", "mean"),
    mean_tau_range=("tau_recorded_range", "mean"),
    mean_peak_to_final_drop=("peak_to_final_drop", "mean"),
    mean_ess_error_change=(
        "ess_absolute_error_change", "mean"
    ),
).reset_index()

condition["auc_cv_pct"] = np.where(
    condition["mean_auc"].abs() > 1e-12,
    100 * condition["sd_auc"]
    / condition["mean_auc"].abs(),
    np.nan
)

condition.to_csv(
    OUT / "condition_diagnostic_summary.csv", index=False
)

# ============================================================
# 4. TEMPERATURE AND ESS UPDATE TRAJECTORIES
# ============================================================

environment_names = [
    "ant-friction",
    "halfcheetah-friction",
    "hopper-friction",
    "walker2d-friction",
]

for env in environment_names:

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for col, shift in enumerate([0.5, 2.0]):

        subset = controller[
            (controller["environment"] == env)
            & np.isclose(controller["shift"], shift)
        ]

        for seed, group in subset.groupby("seed"):

            group = group.sort_values("update_index")

            axes[0, col].plot(
                group["update_index"],
                group["tau"],
                marker="o",
                label=f"Seed {seed}"
            )

            axes[1, col].plot(
                group["update_index"],
                group["ess"],
                marker="o",
                label=f"Seed {seed}"
            )

        axes[0, col].axhline(
            INITIAL_TAU, linestyle="--", color="gray"
        )

        axes[1, col].axhline(
            TARGET_ESS, linestyle="--", color="black",
            label="ESS target"
        )

        axes[0, col].set_title(f"Shift {shift}")
        axes[0, col].set_ylabel("Temperature")
        axes[1, col].set_ylabel("ESS fraction")

        for row in range(2):
            axes[row, col].set_xlabel(
                "Controller update index"
            )
            axes[row, col].grid(alpha=0.3)
            axes[row, col].legend(fontsize=8)

    fig.suptitle(
        f"{env}: adaptive temperature and ESS",
        fontsize=14
    )

    fig.tight_layout()

    fig.savefig(
        OUT / f"{env}_controller_trajectories.png",
        dpi=180
    )

    plt.close(fig)

# ============================================================
# 5. LEARNING CURVES AND LATE-TRAINING PERFORMANCE
# ============================================================

curves = curves.sort_values(KEYS + ["step"])

expected_steps = set(range(1000, 10001, 1000))

for key, group in curves.groupby(KEYS):
    if set(group["step"]) != expected_steps or len(group) != 10:
        raise RuntimeError(
            f"Incomplete or duplicate learning checkpoints: {key}"
        )

learning_records = []

for key, group in curves.groupby(KEYS):

    env, shift, seed = key
    group = group.sort_values("step")

    scores = group["normalized_score"].to_numpy()
    steps = group["step"].to_numpy()

    early = scores[:5]
    late = scores[5:]

    learning_records.append({
        "environment": env,
        "shift": shift,
        "seed": seed,
        "early_mean_score": early.mean(),
        "late_mean_score": late.mean(),
        "late_minus_early_score": (
            late.mean() - early.mean()
        ),
        "final_minus_first_score": (
            scores[-1] - scores[0]
        ),
        "peak_step": int(steps[np.argmax(scores)]),
        "peak_score": float(scores.max()),
        "final_score": float(scores[-1]),
        "final_minus_peak": (
            scores[-1] - scores.max()
        )
    })

learning = pd.DataFrame(learning_records)

learning.to_csv(
    OUT / "learning_stability_diagnostics.csv",
    index=False
)

for env in environment_names:

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for col, shift in enumerate([0.5, 2.0]):

        subset = curves[
            (curves["environment"] == env)
            & np.isclose(curves["shift"], shift)
        ]

        for seed, group in subset.groupby("seed"):

            group = group.sort_values("step")

            axes[col].plot(
                group["step"],
                group["normalized_score"],
                marker="o",
                label=f"Seed {seed}"
            )

        axes[col].set_title(f"Shift {shift}")
        axes[col].set_xlabel(
            "Target-environment interaction step"
        )
        axes[col].set_ylabel("Normalized evaluation score")
        axes[col].grid(alpha=0.3)
        axes[col].legend()

    fig.suptitle(f"{env}: learning curves")
    fig.tight_layout()

    fig.savefig(
        OUT / f"{env}_learning_curves.png",
        dpi=180
    )

    plt.close(fig)

# ============================================================
# 6. EXPLORATORY RUN-LEVEL ASSOCIATIONS
# ============================================================

# Correlations are computed separately by environment.
# They are descriptive only; n=6 per environment.
# We do not infer causation from these associations.

association_rows = []

for env, group in diagnostics.groupby("environment"):

    for variable in [
        "final_tau",
        "mean_ess",
        "ess_mae",
        "tau_recorded_range",
        "peak_to_final_drop"
    ]:

        valid = group[["auc", variable]].dropna()

        if (
            len(valid) >= 4
            and valid["auc"].nunique() > 1
            and valid[variable].nunique() > 1
        ):
            rho = valid["auc"].corr(
                valid[variable], method="spearman"
            )
        else:
            rho = np.nan

        association_rows.append({
            "environment": env,
            "variable": variable,
            "spearman_rho": rho,
            "n": len(valid),
            "interpretation": (
                "Exploratory; shift may confound association"
            )
        })

associations = pd.DataFrame(association_rows)

associations.to_csv(
    OUT / "exploratory_auc_associations.csv",
    index=False
)

# ============================================================
# 7. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 95)
print("FINAL UCDGT CONTROLLER DIAGNOSTICS")
print("=" * 95)

print("\nCONDITION-LEVEL SUMMARY")
print(condition.round(4).to_string(index=False))

print("\nCONTROLLER RUN DIAGNOSTICS")
print(diagnostics.round(4).to_string(index=False))

print("\nLEARNING STABILITY")
print(learning.round(4).to_string(index=False))

print("\nEXPLORATORY ASSOCIATIONS")
print(associations.round(4).to_string(index=False))

print("\nAll diagnostics saved to:")
print(OUT.resolve())

print("\nNOTE:")
print("Controller update indices and evaluation steps")
print("are different time axes and have not been merged.")
print("Associations are descriptive, not causal.")
