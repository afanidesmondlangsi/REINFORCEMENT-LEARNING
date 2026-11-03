import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ============================================================
# RQ3: FIVE-ALGORITHM POST-ADAPTATION COMPARISON
# Interval: 5,000-10,000 steps
# Algorithms: BC-SAC, BC-PAR, RLPD, EA-NTC Normalized, EA-NTC Mean
# SAC Target-Only is intentionally excluded.
# ============================================================

PROJECT = os.getcwd()
EA_ROOT = os.path.join(PROJECT, "experiments_ea_ntc_10k_96")
BASELINE_AUC = os.path.join(PROJECT, "rq3_analysis", "rq3_auc_by_run.csv")
OUT = os.path.join(PROJECT, "rq3_analysis", "RQ3_FIVE_ALGORITHM_5000_10000")
os.makedirs(OUT, exist_ok=True)

TAG = "test/target_normalized_score"
EXPECTED_STEPS = [5000, 6000, 7000, 8000, 9000, 10000]
ENVIRONMENTS = ["ant-friction", "halfcheetah-friction", "hopper-friction", "walker2d-friction"]
SHIFTS = [0.1, 0.5, 2.0, 5.0]
SEEDS = [0, 1, 2]

METHOD_MAP = {
    "BC_SAC": "BC-SAC",
    "BC_PAR": "BC-PAR",
    "RLPD": "RLPD",
    "EA_NTC_NORMALIZED": "EA-NTC Normalized",
    "EA_NTC_UNNORMALIZED": "EA-NTC Mean",
}


def read_ea_ntc_runs():
    """Read all event files, grouping by the EA-NTC run-directory name."""
    if not os.path.isdir(EA_ROOT):
        raise FileNotFoundError(f"EA-NTC directory not found: {EA_ROOT}")

    run_pattern = re.compile(
        r"^(NORMALIZED|UNNORMALIZED)_(ant|halfcheetah|hopper|walker2d)_shift_([0-9.]+)_seed_([0-2])$",
        re.IGNORECASE,
    )
    run_events = {}
    discovered_files = 0

    for current, _, files in os.walk(EA_ROOT):
        for filename in files:
            if not filename.startswith("events.out.tfevents"):
                continue
            path = os.path.join(current, filename)
            discovered_files += 1

            # Find the matching run directory in the current path or its parents.
            matched = None
            for part in reversed(os.path.normpath(current).split(os.sep)):
                m = run_pattern.match(part)
                if m:
                    matched = m
                    break
            if not matched:
                continue

            variant, env, shift, seed = matched.groups()
            method = "EA_NTC_NORMALIZED" if variant.upper() == "NORMALIZED" else "EA_NTC_UNNORMALIZED"
            environment = env.lower() + "-friction"
            key = (method, environment, float(shift), int(seed))
            run_events.setdefault(key, []).append(path)

    print(f"EA-NTC event files discovered: {discovered_files}")
    print(f"EA-NTC run groups discovered: {len(run_events)}")

    rows = []
    problems = []
    for key, paths in sorted(run_events.items()):
        values_by_step = {}
        for path in paths:
            try:
                acc = EventAccumulator(path)
                acc.Reload()
                tags = acc.Tags().get("scalars", [])
                if TAG not in tags:
                    continue
                for event in acc.Scalars(TAG):
                    if 5000 <= event.step <= 10000:
                        step = int(event.step)
                        value = float(event.value)
                        if step in values_by_step and not np.isclose(values_by_step[step], value, rtol=1e-6, atol=1e-8):
                            problems.append((key, step, values_by_step[step], value, path))
                        else:
                            values_by_step[step] = value
            except Exception as exc:
                problems.append((key, "READ_ERROR", str(exc), path))

        missing = sorted(set(EXPECTED_STEPS) - set(values_by_step))
        extras = sorted(set(values_by_step) - set(EXPECTED_STEPS))
        if missing:
            problems.append((key, "MISSING_STEPS", missing, paths))
            continue
        if extras:
            print(f"NOTE: {key} has additional in-range steps: {extras}")

        x = np.array(EXPECTED_STEPS, dtype=float)
        y = np.array([values_by_step[s] for s in EXPECTED_STEPS], dtype=float)
        normalized_auc = float(np.trapz(y, x) / (x[-1] - x[0]))
        raw_auc = float(np.trapz(y, x))
        rows.append({
            "environment": key[1], "shift": key[2], "algorithm": key[0], "seed": key[3],
            "auc_5000_10000": raw_auc,
            "normalized_auc_5000_10000": normalized_auc,
            "checkpoint_count": len(EXPECTED_STEPS),
            "first_step": EXPECTED_STEPS[0], "last_step": EXPECTED_STEPS[-1],
            "source_event_files": len(paths),
        })

    if problems:
        problem_path = os.path.join(OUT, "EA_NTC_extraction_problems.txt")
        with open(problem_path, "w", encoding="utf-8") as f:
            for p in problems:
                f.write(repr(p) + "\n")
        print(f"WARNING: {len(problems)} EA-NTC extraction issues. See {problem_path}")

    ea_df = pd.DataFrame(rows)
    if ea_df.empty:
        raise RuntimeError("No complete EA-NTC post-adaptation runs were extracted.")
    ea_df.to_csv(os.path.join(OUT, "EA_NTC_AUC_by_run_5000_10000.csv"), index=False)
    return ea_df, problems


def load_baselines():
    if not os.path.isfile(BASELINE_AUC):
        raise FileNotFoundError(
            f"Baseline AUC file not found: {BASELINE_AUC}. "
            "Expected rq3_analysis/rq3_auc_by_run.csv from the existing baseline extraction."
        )
    df = pd.read_csv(BASELINE_AUC)
    required = {"environment", "shift", "algorithm", "seed", "normalized_auc_5000_10000"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Baseline AUC CSV is missing columns: {sorted(missing)}")
    df = df[df["algorithm"].isin(["BC_SAC", "BC_PAR", "RLPD"])].copy()
    df["environment"] = df["environment"].astype(str).str.lower()
    df["shift"] = pd.to_numeric(df["shift"])
    df["seed"] = pd.to_numeric(df["seed"]).astype(int)
    df["normalized_auc_5000_10000"] = pd.to_numeric(df["normalized_auc_5000_10000"])
    return df[["environment", "shift", "algorithm", "seed", "normalized_auc_5000_10000"]]


def validate_and_merge(ea_df, baseline_df):
    ea = ea_df[["environment", "shift", "algorithm", "seed", "normalized_auc_5000_10000"]].copy()
    base = baseline_df.copy()
    combined = pd.concat([base, ea], ignore_index=True)
    combined["shift"] = pd.to_numeric(combined["shift"])
    combined["seed"] = pd.to_numeric(combined["seed"]).astype(int)

    duplicate = combined.duplicated(["environment", "shift", "algorithm", "seed"], keep=False)
    if duplicate.any():
        combined.loc[duplicate].sort_values(["algorithm", "environment", "shift", "seed"]).to_csv(
            os.path.join(OUT, "DUPLICATE_RUN_KEYS.csv"), index=False
        )
        raise ValueError("Duplicate environment-shift-algorithm-seed keys found. See DUPLICATE_RUN_KEYS.csv")

    expected_methods = list(METHOD_MAP)
    expected_keys = {
        (env, shift, method, seed)
        for env in ENVIRONMENTS for shift in SHIFTS
        for method in expected_methods for seed in SEEDS
    }
    actual_keys = set(zip(combined["environment"], combined["shift"], combined["algorithm"], combined["seed"]))
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing:
        pd.DataFrame(missing, columns=["environment", "shift", "algorithm", "seed"]).to_csv(
            os.path.join(OUT, "MISSING_RQ3_RUNS.csv"), index=False
        )
        raise ValueError(f"Missing {len(missing)} expected run records. See MISSING_RQ3_RUNS.csv")
    if unexpected:
        print("WARNING: Unexpected run keys:", len(unexpected))

    if len(combined) != 240:
        raise ValueError(f"Expected 240 run records, found {len(combined)}")

    combined["method_label"] = combined["algorithm"].map(METHOD_MAP)
    combined.to_csv(os.path.join(OUT, "RQ3_ALL_FIVE_ALGORITHMS_AUC_by_run.csv"), index=False)
    return combined


def make_outputs(df):
    # Per-condition mean +/- SD across the three matched seeds.
    condition = (df.groupby(["environment", "shift", "algorithm", "method_label"])
                   ["normalized_auc_5000_10000"]
                   .agg(mean="mean", sd="std", n="count")
                   .reset_index())
    condition.to_csv(os.path.join(OUT, "RQ3_condition_level_AUC_mean_sd.csv"), index=False)

    # Overall and environment-level summaries.
    overall = (df.groupby(["algorithm", "method_label"])["normalized_auc_5000_10000"]
                 .agg(runs="count", mean="mean", sd="std")
                 .reset_index())
    overall.to_csv(os.path.join(OUT, "RQ3_overall_algorithm_summary.csv"), index=False)

    env_summary = (df.groupby(["environment", "algorithm", "method_label"])["normalized_auc_5000_10000"]
                     .agg(runs="count", mean="mean", sd="std")
                     .reset_index())
    env_summary.to_csv(os.path.join(OUT, "RQ3_environment_algorithm_summary.csv"), index=False)

    # Count condition wins using the largest mean across three seeds.
    winners = condition.loc[condition.groupby(["environment", "shift"])["mean"].idxmax()]
    wins = winners.groupby(["algorithm", "method_label"]).size().reset_index(name="condition_wins")
    wins.to_csv(os.path.join(OUT, "RQ3_condition_wins.csv"), index=False)

    # LaTeX table: one row per environment-shift, five algorithm columns.
    means = condition.pivot(index=["environment", "shift"], columns="method_label", values="mean")
    sds = condition.pivot(index=["environment", "shift"], columns="method_label", values="sd")
    labels = ["BC-SAC", "BC-PAR", "RLPD", "EA-NTC Normalized", "EA-NTC Mean"]
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{RQ3 post-adaptation time-averaged normalized AUC from 5,000 to 10,000 training steps. Values are mean $\pm$ standard deviation across three seeds. Bold indicates the largest mean within each environment--shift condition.}",
        r"\label{tab:rq3_post_adaptation_auc}", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Environment & Shift & BC-SAC & BC-PAR & RLPD & EA-NTC Norm. & EA-NTC Mean \\",
        r"\midrule"
    ]
    for env in ENVIRONMENTS:
        env_print = env.replace("-friction", "").title()
        for shift in SHIFTS:
            key = (env, shift)
            vals = {lab: means.loc[key, lab] for lab in labels}
            best = max(vals.values())
            cells = []
            for lab in labels:
                m = vals[lab]
                sd = sds.loc[key, lab]
                cell = f"{m:.3f} $\\pm$ {sd:.3f}"
                if np.isclose(m, best):
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
            lines.append(f"{env_print} & {shift:g} & " + " & ".join(cells) + r" \\")
        if env != ENVIRONMENTS[-1]:
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{minipage}{0.98\textwidth}", r"\footnotesize",
        r"\textit{Note:} AUC is computed using the trapezoidal rule over evaluation checkpoints at 5,000, 6,000, 7,000, 8,000, 9,000, and 10,000 steps, then divided by the 5,000-step interval. SAC Target-Only is excluded from this RQ3 algorithm comparison.",
        r"\end{minipage}", r"\end{table*}"
    ]
    tex = "\n".join(lines)
    with open(os.path.join(OUT, "RQ3_FIVE_ALGORITHM_AUC_TABLE.tex"), "w", encoding="utf-8") as f:
        f.write(tex)

    # Overall summary LaTeX table.
    with open(os.path.join(OUT, "RQ3_OVERALL_SUMMARY_TABLE.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join([
            r"\begin{table}[t]", r"\centering",
            r"\caption{Overall RQ3 post-adaptation normalized AUC across 48 runs per algorithm.}",
            r"\label{tab:rq3_overall_auc}", r"\begin{tabular}{lrr}", r"\toprule",
            r"Algorithm & Mean AUC & SD \\", r"\midrule",
            *[f"{r.method_label} & {r['mean']:.3f} & {r['sd']:.3f} " + r"\\" for _, r in overall.iterrows()],
            r"\bottomrule", r"\end{tabular}", r"\end{table}"
        ]))

    # Plot 2x2 environment panels: normalized AUC vs friction shift.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.ravel()
    colors = {
        "BC-SAC": "#1f77b4", "BC-PAR": "#ff7f0e", "RLPD": "#2ca02c",
        "EA-NTC Normalized": "#d62728", "EA-NTC Mean": "#9467bd"
    }
    for ax, env in zip(axes, ENVIRONMENTS):
        sub = condition[condition.environment == env]
        for lab in labels:
            s = sub[sub.method_label == lab].sort_values("shift")
            ax.errorbar(s["shift"], s["mean"], yerr=s["sd"], marker="o", capsize=3,
                        linewidth=1.8, label=lab, color=colors[lab])
        ax.set_title(env.replace("-friction", "").title())
        ax.set_xlabel("Friction shift")
        ax.set_ylabel("Time-averaged normalized AUC")
        ax.set_xticks(SHIFTS)
        ax.grid(True, alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("RQ3: Post-adaptation performance across environments and friction shifts", y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(os.path.join(OUT, "RQ3_five_algorithm_by_environment.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "RQ3_five_algorithm_by_environment.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Overall algorithm means bar plot.
    order = labels
    overall = overall.set_index("method_label").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(overall["method_label"], overall["mean"], yerr=overall["sd"], capsize=5)
    ax.set_ylabel("Time-averaged normalized AUC (5K–10K)")
    ax.set_title("Overall RQ3 post-adaptation performance")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "RQ3_five_algorithm_overall.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "RQ3_five_algorithm_overall.pdf"), bbox_inches="tight")
    plt.close(fig)

    print("\n========== RQ3 FIVE-ALGORITHM SUMMARY ==========")
    print(overall.to_string(index=False))
    print("\n========== CONDITION WINS ==========")
    print(wins.to_string(index=False))
    print("\n========== ENVIRONMENT SUMMARY ==========")
    print(env_summary.to_string(index=False))


def main():
    print("========== EXTRACTING EA-NTC 5K-10K AUC ==========")
    ea_df, problems = read_ea_ntc_runs()
    print("EA-NTC complete runs extracted:", len(ea_df))
    print(ea_df.groupby("algorithm").size().to_string())

    print("\n========== LOADING EXISTING BASELINES ==========")
    baseline_df = load_baselines()
    print(baseline_df.groupby("algorithm").size().to_string())

    combined = validate_and_merge(ea_df, baseline_df)
    print("\nVerified total five-algorithm runs:", len(combined))
    print(combined.groupby("algorithm").size().to_string())
    make_outputs(combined)
    print("\nRQ3 five-algorithm analysis completed.")
    print("Output directory:", OUT)
    print("SAC Target-Only was excluded as requested.")

if __name__ == "__main__":
    main()
