import numpy as np
from envs.mujoco.call_mujoco_env import call_mujoco_env

def make_env(shift):
    env = call_mujoco_env({
        "env_name": "hopper-friction",
        "shift_level": shift,
    })
    env.seed(123)
    return env

a = make_env(0.1)
b = make_env(0.5)

try:
    a.reset()
    b.reset()

    rng = np.random.RandomState(999)

    print("STEP | CONTACTS 0.1 | CONTACTS 0.5 | MAX CONTACT FRICTION DIFF")

    contact_steps = 0
    friction_diff_steps = 0

    for step in range(100):
        action = rng.uniform(
            -0.5, 0.5,
            size=a.action_space.shape
        )

        _, _, done_a, _ = a.step(action)
        _, _, done_b, _ = b.step(action)

        sim_a = a.unwrapped.sim
        sim_b = b.unwrapped.sim

        contacts_a = sim_a.data.ncon
        contacts_b = sim_b.data.ncon

        if contacts_a or contacts_b:
            contact_steps += 1

        fa = [
            np.array(sim_a.data.contact[i].friction)
            for i in range(contacts_a)
        ]

        fb = [
            np.array(sim_b.data.contact[i].friction)
            for i in range(contacts_b)
        ]

        if len(fa) == len(fb) and fa:
            max_diff = max(
                float(np.max(np.abs(x - y)))
                for x, y in zip(fa, fb)
            )
        else:
            max_diff = float("nan")

        if np.isfinite(max_diff) and max_diff > 0:
            friction_diff_steps += 1

        if step < 10 or step % 10 == 9:
            print(
                f"{step+1:4d} | "
                f"{contacts_a:12d} | "
                f"{contacts_b:12d} | "
                f"{max_diff:.8f}"
            )

        if done_a or done_b:
            break

    print("\nContact-active steps:", contact_steps)
    print("Steps with different contact friction:", friction_diff_steps)

finally:
    a.close()
    b.close()
