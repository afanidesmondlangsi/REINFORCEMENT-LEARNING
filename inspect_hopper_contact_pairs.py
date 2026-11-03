import numpy as np
from collections import Counter
from envs.mujoco.call_mujoco_env import call_mujoco_env

def make_env(shift):
    env = call_mujoco_env({
        "env_name": "hopper-friction",
        "shift_level": shift
    })
    env.seed(123)
    env.reset()
    return env

envs = {
    "0.1": make_env(0.1),
    "0.5": make_env(0.5)
}

rng = np.random.RandomState(999)
pairs = {"0.1": Counter(), "0.5": Counter()}
examples = set()

try:
    for step in range(100):
        action = rng.uniform(
            -0.5, 0.5,
            size=envs["0.1"].action_space.shape
        )

        done_flags = []

        for shift, env in envs.items():
            _, _, done, _ = env.step(action)
            done_flags.append(done)

            sim = env.unwrapped.sim
            model = sim.model
            data = sim.data

            for i in range(data.ncon):
                contact = data.contact[i]

                name1 = model.geom_id2name(contact.geom1)
                name2 = model.geom_id2name(contact.geom2)

                pair = tuple(sorted((name1, name2)))
                pairs[shift][pair] += 1

                key = (shift, pair)

                if key not in examples:
                    examples.add(key)

                    print(
                        f"Shift={shift}, step={step+1}, "
                        f"contact={pair}, "
                        f"friction={np.array(contact.friction)}"
                    )

        if any(done_flags):
            break

    print("\n========== CONTACT PAIR COUNTS ==========")

    for shift, counts in pairs.items():
        print(f"\nShift {shift}")

        for pair, count in counts.most_common():
            print(f"  {pair}: {count}")

finally:
    for env in envs.values():
        env.close()
