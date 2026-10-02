from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT =  ROOT = Path("rq1_generalization")

TAG = "test/target_normalized_score"

valid_dirs = []

# ------------------------------------------------------------
# Find HalfCheetah TensorBoard directories
# ------------------------------------------------------------

for tb_dir in ROOT.rglob("tb"):

    path_text = str(tb_dir).replace("\\", "/")

    if "halfcheetah" not in path_text.lower():
        continue

    # Exclude the old incorrect shift=2 directories.
    # We only want 0.5 and 2.0.
    parent_parts = tb_dir.parts

    if any(
        part in {
            "halfcheetah-friction-2",
            "halfcheetah-friction-srcdatatype-medium-2",
        }
        for part in parent_parts
    ):
        continue

    valid_dirs.append(tb_dir)


print("\n============================================================")
print("HALFCHEETAH RQ1 TENSORBOARD VERIFICATION")
print("============================================================\n")


valid_runs = []


for tb_dir in sorted(valid_dirs):

    event_files = sorted(
        tb_dir.glob("events.out.tfevents*"),
        key=lambda x: x.stat().st_mtime
    )

    if not event_files:
        continue

    # --------------------------------------------------------
    # Examine each event file independently.
    # Keep files containing our required scalar.
    # --------------------------------------------------------

    candidates = []

    for event_file in event_files:

        try:
            ea = EventAccumulator(
                str(event_file),
                size_guidance={"scalars": 0}
            )

            ea.Reload()

            scalar_tags = ea.Tags().get("scalars", [])

            if TAG not in scalar_tags:
                continue

            events = ea.Scalars(TAG)

            steps = [e.step for e in events]

            candidates.append({
                "file": event_file,
                "steps": steps,
                "events": events,
            })

        except Exception as e:
            print("Could not read:", event_file)
            print("Reason:", e)


    # --------------------------------------------------------
    # Require a complete experiment.
    # --------------------------------------------------------

    complete = []

    for candidate in candidates:

        steps = candidate["steps"]

        if 5000 in steps:
            complete.append(candidate)


    if not complete:

        print("INCOMPLETE:")
        print(tb_dir)
        print()
        continue


    # --------------------------------------------------------
    # If the experiment was repeated, use the most recent
    # complete TensorBoard event file.
    # --------------------------------------------------------

    selected = max(
        complete,
        key=lambda x: x["file"].stat().st_mtime
    )

    event_file = selected["file"]
    events = selected["events"]

    print("VALID RUN")
    print("Directory :", tb_dir)
    print("Event file:", event_file.name)

    print("Scores:")

    for event in events:
        if event.step in [1000, 2000, 3000, 4000, 5000]:
            print(
                f"  step={event.step:4d} "
                f"score={event.value:.6f}"
            )

    print()

    valid_runs.append({
        "directory": str(tb_dir),
        "event_file": str(event_file),
    })


print("============================================================")
print("TOTAL VALID RUNS:", len(valid_runs))
print("============================================================")