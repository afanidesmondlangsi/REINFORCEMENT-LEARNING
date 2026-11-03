import numpy as np
from envs.mujoco.call_mujoco_env import call_mujoco_env

def create_env(shift):
    env = call_mujoco_env({
        "env_name": "hopper-friction",
        "shift_level": shift,
    })
    env.seed(123)
    return env

env_a = create_env(0.1)
env_b = create_env(0.5)

try:
    model_a = env_a.unwrapped.sim.model
    model_b = env_b.unwrapped.sim.model

    print("\n========== LOADED FRICTION ==========")

    for name in ["floor", "torso_geom", "thigh_geom",
                 "leg_geom", "foot_geom"]:

        id_a = model_a.geom_name2id(name)
        id_b = model_b.geom_name2id(name)

        fa = model_a.geom_friction[id_a]
        fb = model_b.geom_friction[id_b]

        print(f"{name:12s} 0.1={fa}  0.5={fb}")

    print("\n========== MATCHED TRANSITIONS ==========")

    obs_a = env_a.reset()
    obs_b = env_b.reset()

    print(
        "Initial observations equal:",
        np.allclose(obs_a, obs_b, rtol=0, atol=0)
    )

    rng = np.random.RandomState(999)

    max_obs_diff = 0.0
    max_reward_diff = 0.0

    for step in range(100):
        action = rng.uniform(
            low=-0.5,
            high=0.5,
            size=env_a.action_space.shape
        )

        next_a, reward_a, done_a, _ = env_a.step(action)
        next_b, reward_b, done_b, _ = env_b.step(action)

        obs_diff = float(
            np.max(np.abs(next_a - next_b))
        )
        reward_diff = abs(reward_a - reward_b)

        max_obs_diff = max(max_obs_diff, obs_diff)
        max_reward_diff = max(max_reward_diff, reward_diff)

        if step in [0, 1, 2, 4, 9, 19, 49, 99]:
            print(
                f"Step {step + 1:3d}: "
                f"obs_diff={obs_diff:.12f}, "
                f"reward_diff={reward_diff:.12f}"
            )

        if done_a or done_b:
            print(
                "Episode ended at step",
                step + 1,
                "done_a=", done_a,
                "done_b=", done_b
            )
            break

    print("\nMaximum observation difference:", max_obs_diff)
    print("Maximum reward difference:", max_reward_diff)

finally:
    env_a.close()
    env_b.close()
