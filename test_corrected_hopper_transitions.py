import numpy as np
from pathlib import Path
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.wrappers.time_limit import TimeLimit

assets = Path("envs/mujoco/assets_corrected").resolve()

def make_env(shift, seed):
    env = TimeLimit(
        HopperEnv(
            xml_file=str(
                assets / f"hopper_friction_{shift}.xml"
            )
        ),
        max_episode_steps=1000
    )
    env.seed(seed)
    return env

shifts = [0.1, 0.5, 2.0, 5.0]
seeds = [0, 1, 2, 3, 4, 5]

print("seed,shift,max_obs_diff,max_reward_diff,steps")

for seed in seeds:

    for shift in shifts[1:]:

        a = make_env(0.1, seed)
        b = make_env(shift, seed)

        try:
            obs_a = a.reset()
            obs_b = b.reset()

            assert np.array_equal(obs_a, obs_b)

            rng = np.random.RandomState(seed + 999)

            max_obs_diff = 0.0
            max_reward_diff = 0.0

            steps = 0

            for step in range(300):

                action = rng.uniform(
                    -0.5,
                    0.5,
                    size=a.action_space.shape
                )

                obs_a, reward_a, done_a, _ = a.step(action)
                obs_b, reward_b, done_b, _ = b.step(action)

                max_obs_diff = max(
                    max_obs_diff,
                    float(np.max(np.abs(obs_a - obs_b)))
                )

                max_reward_diff = max(
                    max_reward_diff,
                    abs(float(reward_a - reward_b))
                )

                steps = step + 1

                if done_a or done_b:
                    break

            print(
                f"{seed},{shift},"
                f"{max_obs_diff:.12g},"
                f"{max_reward_diff:.12g},"
                f"{steps}"
            )

        finally:
            a.close()
            b.close()
