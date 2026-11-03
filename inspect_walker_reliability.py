from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path("logs_ea_ntc_norm_vs_mean_4env_6shift_6seed_5000")

for treatment in ("NORMALIZED", "UNNORMALIZED"):
    run = root / f"{treatment}_walker2d_shift_0.5_seed_0"
    files = list(run.rglob("events.out.tfevents.*"))

    if not files:
        print("Missing events:", run)
        continue

    ea = EventAccumulator(
        str(files[0].parent),
        size_guidance={"scalars": 0}
    )
    ea.Reload()

    print("\nTREATMENT:", treatment)

    for tag in [
        "ea_ntc/heldout_relative_mse",
        "ea_ntc/heldout_target_mse",
        "ea_ntc/heldout_zero_delta_mse",
        "ea_ntc/target_calibration_count",
        "ea_ntc/target_dynamics_train_count",
        "ea_ntc/target_uncertainty_reference",
    ]:
        if tag not in ea.Tags().get("scalars", []):
            print(tag, "NOT LOGGED")
            continue

        values = ea.Scalars(tag)

        if values:
            print(
                tag,
                "first =", values[0].value,
                "last =", values[-1].value,
                "min =", min(v.value for v in values),
                "max =", max(v.value for v in values)
            )
