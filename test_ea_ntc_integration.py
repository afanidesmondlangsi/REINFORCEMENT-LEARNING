
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, "./algo")

from call_algo import call_algo
from utils import ReplayBuffer


def make_transition(state_dim, action_dim):
    state = np.random.randn(state_dim).astype(np.float32)
    action = np.random.uniform(
        -1, 1, size=action_dim
    ).astype(np.float32)

    delta = np.zeros(state_dim, dtype=np.float32)
    delta[:action_dim] = 0.1 * action
    delta += 0.01 * state

    next_state = state + delta
    reward = -float(np.sum(action ** 2))

    return state, action, next_state, reward, 0.0


def populate(buffer, count, state_dim, action_dim):
    for _ in range(count):
        buffer.add(
            *make_transition(state_dim, action_dim)
        )


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)

    with open(
        "config/mujoco/ea_ntc_bc_sac/halfcheetah.yaml",
        "r"
    ) as f:
        config = yaml.safe_load(f)

    # Smaller networks and earlier warm-up
    # for a fast integration smoke test.
    config["hidden_sizes"] = 32
    config["ea_hidden_size"] = 32
    config["ea_ensemble_size"] = 3
    config["ea_warmup_target_steps"] = 32
    config["ea_min_dynamics_updates"] = 3
    config["ea_max_calibration_error"] = 1.0
    config["ea_dynamics_batch_size"] = 16
    config["ea_dynamics_update_interval"] = 2
    config["ea_calibration_update_interval"] = 2

    device = torch.device("cpu")

    policy = call_algo(
        "ea_ntc_bc_sac",
        config,
        1,
        device,
    )

    state_dim = config["state_dim"]
    action_dim = config["action_dim"]

    src_buffer = ReplayBuffer(
        state_dim, action_dim, device,
        max_size=256
    )

    tar_buffer = ReplayBuffer(
        state_dim, action_dim, device,
        max_size=256
    )

    populate(src_buffer, 128, state_dim, action_dim)

    # Before warm-up: no calibration should occur.
    populate(tar_buffer, 16, state_dim, action_dim)

    policy.train(
        src_buffer, tar_buffer,
        batch_size=16
    )

    assert not policy.ea_gate_ready
    print("PASS: Gate inactive before warm-up.")

    # Complete target warm-up.
    populate(tar_buffer, 48, state_dim, action_dim)

    for _ in range(12):
        policy.train(
            src_buffer, tar_buffer,
            batch_size=16
        )

    assert policy.ea_dynamics_updates > 0
    print("PASS: Dynamics ensemble updated.")

    assert policy.ea_gate_ready
    print("PASS: Calibration gate activated.")

    assert policy.ea_last_weights is not None
    assert torch.isfinite(
        policy.ea_last_weights
    ).all()

    assert torch.all(
        policy.ea_last_weights >= 0
    )
    assert torch.all(
        policy.ea_last_weights <= 1
    )

    print("PASS: Source weights are finite and bounded.")

    for parameter in policy.policy.parameters():
        assert torch.isfinite(parameter).all()

    for parameter in policy.q_funcs.parameters():
        assert torch.isfinite(parameter).all()

    print("PASS: Actor and critic parameters are finite.")

    print(
        "Dynamics updates:",
        policy.ea_dynamics_updates
    )

    print(
        "Target reference:",
        policy.calibrator.target_reference
    )

    print(
        "Mean source weight:",
        policy.ea_last_weights.mean().item()
    )

    print(
        "Mean ensemble uncertainty:",
        policy.ea_last_uncertainty.mean().item()
    )

    print("\nEA-NTC INTEGRATION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

