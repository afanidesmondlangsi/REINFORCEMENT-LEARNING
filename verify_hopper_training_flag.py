import numpy as np
from envs.mujoco.call_mujoco_env import call_mujoco_env

expected_floor = {
    False: 1.0,
    True: 0.01,
}

expected_foot = {
    0.1: 0.2,
    0.5: 1.0,
    2.0: 4.0,
    5.0: 10.0,
}

for corrected in [False, True]:

    print("\n====================================")
    print(f"Corrected Hopper: {corrected}")
    print("====================================")

    for shift in [0.1, 0.5, 2.0, 5.0]:

        env = call_mujoco_env({
            "env_name": "hopper-friction",
            "shift_level": shift,
            "corrected_hopper_friction": corrected,
        })

        try:
            model = env.unwrapped.sim.model

            floor_id = model.geom_name2id("floor")
            foot_id = model.geom_name2id("foot_geom")

            floor_friction = float(
                model.geom_friction[floor_id][0]
            )

            foot_friction = float(
                model.geom_friction[foot_id][0]
            )

            assert np.isclose(
                floor_friction,
                expected_floor[corrected]
            ), "Incorrect floor friction"

            assert np.isclose(
                foot_friction,
                expected_foot[shift]
            ), "Incorrect foot friction"

            print(
                f"Shift {shift}: "
                f"floor={floor_friction:.2f}, "
                f"foot={foot_friction:.2f} PASS"
            )

        finally:
            env.close()

print("\nPASS: Original and corrected environments load correctly.")
