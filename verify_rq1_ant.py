from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ============================================================
# RQ1 ant TENSORBOARD VERIFICATION
# ============================================================

ROOT = Path("rq1_generalization")

TAG = "test/target_normalized_score"

REQUIRED_STEPS = [1000, 2000, 3000, 4000, 5000]

EXPECTED_RUNS = 9


# ============================================================
# FIND ant RQ1 DIRECTORIES
# ============================================================

valid_dirs = []

for tb_dir in ROOT.rglob("tb"):

    path_text = str(tb_dir).replace("\\", "/").lower()

    # Only ant
    if "ant" not in path_text:
        continue

    # Only RQ1 shift = 0.1
    #
    # Expected directory patterns:
    #
    # SAC_TARGET_ONLY:
    # ant-friction-0.1
    #
    # BC_SAC / RLPD:
    # ant-friction-srcdatatype-medium-0.1
    #
    if (
        "ant-friction-0.1" not in path_text
        and
        "ant-friction-srcdatatype-medium-0.1" not in path_text
    ):
        continue

    valid_dirs.append(tb_dir)


# ============================================================
# VERIFY RUNS
# ============================================================

print()
print("============================================================")
print("ant RQ1 TENSORBOARD VERIFICATION")
print("============================================================")
print()


valid_runs = []


for tb_dir in sorted(valid_dirs):

    event_files = sorted(
        tb_dir.glob("events.out.tfevents*"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not event_files:
        continue


    # --------------------------------------------------------
    # Find the newest event file containing all required steps
    # --------------------------------------------------------

    selected_file = None
    selected_scores = None


    for event_file in event_files:

        try:

            ea = EventAccumulator(str(event_file))
            ea.Reload()

            if TAG not in ea.Tags().get("scalars", []):
                continue

            events = ea.Scalars(TAG)

            score_by_step = {
                int(event.step): float(event.value)
                for event in events
            }

            if all(
                step in score_by_step
                for step in REQUIRED_STEPS
            ):

                selected_file = event_file

                selected_scores = {
                    step: score_by_step[step]
                    for step in REQUIRED_STEPS
                }

                break

        except Exception as e:

            print(f"Could not read: {event_file}")
            print(f"Reason: {e}")


    # --------------------------------------------------------
    # Print valid run
    # --------------------------------------------------------

    if selected_file is not None:

        valid_runs.append(tb_dir)

        print("VALID RUN")
        print(f"Directory : {tb_dir}")
        print(f"Event file: {selected_file.name}")
        print("Scores:")

        for step in REQUIRED_STEPS:

            print(
                f"  step={step} "
                f"score={selected_scores[step]:.6f}"
            )

        print()


# ============================================================
# FINAL CHECK
# ============================================================

print("============================================================")
print(f"TOTAL VALID RUNS: {len(valid_runs)}")
print("============================================================")


if len(valid_runs) == EXPECTED_RUNS:

    print()
    print("VERIFICATION PASSED")
    print(
        f"All {EXPECTED_RUNS} ant RQ1 runs contain "
        "the required evaluation steps."
    )

else:

    print()
    print("VERIFICATION FAILED")
    print(
        f"Expected {EXPECTED_RUNS} valid runs, "
        f"but found {len(valid_runs)}."
    )
