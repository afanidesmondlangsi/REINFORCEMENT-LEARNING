from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path("logs_walker2d_gate_sensitivity_090_095")

for treatment in ("NORMALIZED", "UNNORMALIZED"):

    run = root / f"{treatment}_walker2d_shift_0.5_seed_0"
    files = list(run.rglob("events.out.tfevents.*"))

    print("\n" + "=" * 60)
    print("TREATMENT:", treatment)

    if not files:
        print("ERROR: No TensorBoard event file")
        continue

    ea = EventAccumulator(
        str(files[0].parent),
        size_guidance={"scalars": 0},
    )
    ea.Reload()

    for tag in (
        "ea_ntc/gate_ready",
        "ea_ntc/mean_source_weight",
        "ea_ntc/adaptive_gate_strength",
        "ea_ntc/heldout_relative_mse",
        "test/target_normalized_score",
    ):
        if tag not in ea.Tags().get("scalars", []):
            print(tag, "NOT LOGGED")
            continue

        values = [v.value for v in ea.Scalars(tag)]

        if values:
            print(
                tag,
                "min =", min(values),
                "max =", max(values),
                "last =", values[-1],
            )

    if "ea_ntc/gate_ready" in ea.Tags().get("scalars", []):
        gate_values = [
            v.value for v in ea.Scalars("ea_ntc/gate_ready")
        ]
        print("GATE EVER ACTIVE:", any(v > 0.5 for v in gate_values))

