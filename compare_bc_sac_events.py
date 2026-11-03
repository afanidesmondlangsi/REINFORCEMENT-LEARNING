from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

dirs = [
    Path(r"experiments_10k_576\results\BC_SAC\bc_sac\hopper-friction-srcdatatype-medium-0.1\r0\tb"),
    Path(r"rq1_test\BC_SAC\hopper-friction-srcdatatype-medium-0.1\r0\tb")
]

for directory in dirs:
    print("\n" + "=" * 65)
    print(directory)

    files = list(directory.glob("events.out.tfevents.*"))

    for file in files:
        ea = EventAccumulator(str(file))
        ea.Reload()

        print("\nEvent file:", file.name)
        print("Scalar tags:", ea.Tags()["scalars"])

        for tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            print("\nTAG:", tag)

            for event in events:
                print("Step:", event.step, "Value:", event.value)
