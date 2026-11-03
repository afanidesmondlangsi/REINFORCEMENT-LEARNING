import numpy as np
from envs.mujoco.call_mujoco_env import call_mujoco_env

for shift in [0.1, 0.5, 2.0, 5.0]:
    env = call_mujoco_env({
        "env_name": "hopper-friction",
        "shift_level": shift
    })
    env.seed(123)
    env.reset()

    sim = env.unwrapped.sim
    model = sim.model

    floor_id = model.geom_name2id("floor")
    foot_id = model.geom_name2id("foot_geom")

    print(f"\nSHIFT {shift}")
    print("Floor friction:", model.geom_friction[floor_id])
    print("Foot friction:", model.geom_friction[foot_id])
    print("Floor priority:", model.geom_priority[floor_id])
    print("Foot priority:", model.geom_priority[foot_id])
    print("Floor condim:", model.geom_condim[floor_id])
    print("Foot condim:", model.geom_condim[foot_id])

    rng = np.random.RandomState(999)

    for step in range(100):
        action = rng.uniform(-0.5, 0.5, env.action_space.shape)
        _, _, done, _ = env.step(action)

        if sim.data.ncon:
            for i in range(sim.data.ncon):
                contact = sim.data.contact[i]
                names = {
                    model.geom_id2name(contact.geom1),
                    model.geom_id2name(contact.geom2)
                }

                if names == {"floor", "foot_geom"}:
                    print("Resolved contact dimension:", contact.dim)
                    print("Effective friction:", np.array(contact.friction))
                    break
            else:
                if done:
                    break
                continue
            break

        if done:
            break

    env.close()
