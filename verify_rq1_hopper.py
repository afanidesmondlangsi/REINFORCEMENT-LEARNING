from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ============================================================
# RQ1 HOPPER TENSORBOARD VERIFICATION
# ============================================================

TAG = "test/target_normalized_score"

REQUIRED_STEPS = [1000, 2000, 3000, 4000, 5000]

# Explicit locations of the nine expected runs
RUNS = [
    # --------------------------------------------------------
    # SAC_TARGET_ONLY
    # --------------------------------------------------------
    (
        "SAC_TARGET_ONLY",
        0,
        Path(
            "rq1_target_only/SAC_TARGET_ONLY/"
            "hopper-friction-0.1/r0/tb"
        ),
    ),
    (
        "SAC_TARGET_ONLY",
        1,
        Path(
            "rq1_target_only/SAC_TARGET_ONLY/"
            "hopper-friction-0.1/r1/tb"
        ),
    ),
    (
        "SAC_TARGET_ONLY",
        2,
        Path(
            "rq1_target_only/SAC_TARGET_ONLY/"
            "hopper-friction-0.1/r2/tb"
        ),
    ),

    # --------------------------------------------------------
    # BC_SAC
    # seed 0 was stored in rq1_test
    # seeds 1 and 2 were stored in rq1_bc_sac
    # --------------------------------------------------------
    (
        "BC_SAC",
        0,
        Path(
            "rq1_test/BC_SAC/"
            "hopper-friction-srcdatatype-medium-0.1/r0/tb"
        ),
    ),
    (
        "BC_SAC",
        1,
        Path(
            "rq1_bc_sac/BC_SAC/"
            "hopper-friction-srcdatatype-medium-0.1/r1/tb"
        ),
    ),
    (
        "BC_SAC",
        2,
        Path(
            "rq1_bc_sac/BC_SAC/"
            "hopper-friction-srcdatatype-medium-0.1/r2/tb"
        ),
    ),

    # --------------------------------------------------------
    # RLPD
    # seed 0 was stored in rq1_rlpd_test
    # seeds 1 and 2 were stored in rq1_rlpd
    # --------------------------------------------------------
    (
        "RLPD",
        0,
        Path(
            "rq1_rlpd_test/RLPD/"
            "hopper-friction-srcdatatype-medium-0.1/r0/tb"
        ),
    ),
    (
        "RLPD",
        1,
        Path(
            "rq1_rlpd/RLPD/"
            "hopper-friction-srcdatatype-medium-0.1/r1/tb"
        ),
    ),
    (
        "RLPD",
        2,
        Path(
            "rq1_rlpd/RLPD/"
            "hopper-friction-srcdatatype-medium-0.1/r2/tb"
        ),
    ),
]


# ============================================================
# VERIFY EACH RUN
# ============================================================

valid_runs = 0

print()
print("=" * 70)
print("RQ1 HOPPER TENSORBOARD VERIFICATION")
print("=" * 70)

for algorithm, seed, tb_dir in RUNS:

    print()
    print("-" * 70)
    print(f"Algorithm : {algorithm}")
    print(f"Seed      : {seed}")
    print(f"Directory : {tb_dir}")
    print("-" * 70)

    # --------------------------------------------------------
    # Check directory
    # --------------------------------------------------------

    if not tb_dir.exists():
        print("INVALID RUN")
        print("Reason: TensorBoard directory does not exist.")
        continue

    # --------------------------------------------------------
    # Find event files
    # --------------------------------------------------------

    event_files = list(tb_dir.glob("events.out.tfevents.*"))

    if not event_files:
        print("INVALID RUN")
        print("Reason: No TensorBoard event file found.")
        continue

    # If multiple files exist, use newest one
    event_file = max(
        event_files,
        key=lambda p: p.stat().st_mtime
    )

    print(f"Event file: {event_file.name}")

    # --------------------------------------------------------
    # Load TensorBoard event file
    # --------------------------------------------------------

    try:
        ea = EventAccumulator(str(event_file))
        ea.Reload()
    except Exception as error:
        print("INVALID RUN")
        print(f"Reason: Could not read event file: {error}")
        continue

    # --------------------------------------------------------
    # Check tag
    # --------------------------------------------------------

    scalar_tags = ea.Tags().get("scalars", [])

    if TAG not in scalar_tags:
        print("INVALID RUN")
        print(f"Reason: Missing TensorBoard tag: {TAG}")
        print(f"Available scalar tags: {scalar_tags}")
        continue

    # --------------------------------------------------------
    # Extract scores
    # --------------------------------------------------------

    events = ea.Scalars(TAG)

    step_to_score = {}

    for event in events:
        step_to_score[int(event.step)] = float(event.value)

    available_steps = sorted(step_to_score.keys())

    print(f"Available steps: {available_steps}")

    # --------------------------------------------------------
    # Check required steps
    # --------------------------------------------------------

    missing_steps = [
        step
        for step in REQUIRED_STEPS
        if step not in step_to_score
    ]

    if missing_steps:
        print("INVALID RUN")
        print(f"Reason: Missing required steps: {missing_steps}")
        continue

    # --------------------------------------------------------
    # Valid run
    # --------------------------------------------------------

    print()
    print("VALID RUN")
    print("Scores:")

    for step in REQUIRED_STEPS:
        print(
            f"  step={step:<4} "
            f"score={step_to_score[step]:.6f}"
        )

    valid_runs += 1


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print(f"TOTAL VALID RUNS: {valid_runs}")
print("=" * 70)

if valid_runs == 9:
    print()
    print("VERIFICATION PASSED")
    print(
        "All 9 Hopper RQ1 runs contain the required "
        "evaluation steps."
    )
else:
    print()
    print("VERIFICATION FAILED")
    print(
        f"Expected 9 complete runs, but found {valid_runs}."
    )