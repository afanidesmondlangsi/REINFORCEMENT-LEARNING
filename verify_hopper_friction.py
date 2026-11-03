from envs.mujoco.call_mujoco_env import call_mujoco_env
import numpy as np

for shift in (0.1, 0.5):
    env = call_mujoco_env({
        "env_name": "hopper-friction",
        "shift_level": shift,
    })

    model = env.unwrapped.model
    friction = np.asarray(model.geom_friction).copy()

    print(f"\nHopper shift {shift}")
    print("Number of geoms:", len(friction))
    print("Geom friction:")
    print(friction)

    env.close()
