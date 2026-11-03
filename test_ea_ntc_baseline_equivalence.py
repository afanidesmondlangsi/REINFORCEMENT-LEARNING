
import sys
import random
import numpy as np
import torch
import yaml

sys.path.insert(0, "./algo")

from call_algo import call_algo
from utils import ReplayBuffer


def populate(buffer, count, state_dim, action_dim):
    for _ in range(count):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.uniform(
            -1, 1, action_dim
        ).astype(np.float32)

        delta = 0.01 * state
        delta[:action_dim] += 0.1 * action

        next_state = state + delta
        reward = -float(np.sum(action ** 2))

        buffer.add(
            state, action, next_state, reward, 0.0
        )


def maximum_difference(model_a, model_b):
    return max(
        (
            (p_a.detach() - p_b.detach())
            .abs().max().item()
        )
        for p_a, p_b in zip(
            model_a.parameters(),
            model_b.parameters()
        )
    )


def main():
    torch.set_num_threads(2)

    seed = 123

    with open(
        "config/mujoco/bc_sac/halfcheetah.yaml"
    ) as f:
        baseline_config = yaml.safe_load(f)

    with open(
        "config/mujoco/ea_ntc_bc_sac/halfcheetah.yaml"
    ) as f:
        ea_config = yaml.safe_load(f)

    baseline_config["hidden_sizes"] = 32
    ea_config["hidden_sizes"] = 32
    ea_config["ea_hidden_size"] = 32
    ea_config["ea_gate_bc"] = False

    device = torch.device("cpu")

    # Reset initialization RNG before each model.
    torch.manual_seed(seed)
    baseline = call_algo(
        "bc_sac", baseline_config, 1, device
    )

    torch.manual_seed(seed)
    ea_ntc = call_algo(
        "ea_ntc_bc_sac", ea_config, 1, device
    )

    assert maximum_difference(
        baseline.policy, ea_ntc.policy
    ) == 0.0

    assert maximum_difference(
        baseline.q_funcs, ea_ntc.q_funcs
    ) == 0.0

    print("PASS: Identical actor and critic initialization.")

    state_dim = baseline_config["state_dim"]
    action_dim = baseline_config["action_dim"]

    src = ReplayBuffer(
        state_dim, action_dim, device, max_size=512
    )
    tar = ReplayBuffer(
        state_dim, action_dim, device, max_size=512
    )

    np.random.seed(seed)
    populate(src, 256, state_dim, action_dim)
    populate(tar, 256, state_dim, action_dim)

    for step in range(5):
        # Save the full RNG state so that both
        # algorithms sample identical replay batches
        # and stochastic policy actions.
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        python_state = random.getstate()

        baseline.train(src, tar, batch_size=32)

        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        random.setstate(python_state)

        ea_ntc.train(src, tar, batch_size=32)

        actor_diff = maximum_difference(
            baseline.policy, ea_ntc.policy
        )

        critic_diff = maximum_difference(
            baseline.q_funcs, ea_ntc.q_funcs
        )

        target_diff = maximum_difference(
            baseline.target_q_funcs,
            ea_ntc.target_q_funcs
        )

        print(
            "Step {}: actor={:.10f}, critic={:.10f}, "
            "target={:.10f}".format(
                step + 1,
                actor_diff,
                critic_diff,
                target_diff,
            )
        )

        assert actor_diff < 1e-6
        assert critic_diff < 1e-6
        assert target_diff < 1e-6

    print(
        "\nPASS: EA-NTC with gating disabled "
        "matches original BC-SAC."
    )


if __name__ == "__main__":
    main()
