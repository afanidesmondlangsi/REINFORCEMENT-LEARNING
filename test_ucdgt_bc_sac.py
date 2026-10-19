
import numpy as np
import torch

from algo.utils import ReplayBuffer
from algo.offline_online.ucdgt_bc_sac import UCDGTBCSAC


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cpu")

    config = {
        "state_dim": 5,
        "action_dim": 2,
        "max_action": 1.0,
        "hidden_sizes": 32,
        "gamma": 0.99,
        "tau": 0.005,
        "update_interval": 2,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "temperature_opt": False,
        "weight": 2.5,

        "ucdgt_warmup": 64,
        "ucdgt_train_steps": 10,
        "ucdgt_cal_fraction": 0.2,
        "ucdgt_ensemble_size": 3,
        "ucdgt_dynamics_lr": 1e-3,
        "ucdgt_gate_tau": 10.0,
        "ucdgt_source_scale": 1.0,
        "ucdgt_gate_actor": True,
    }

    policy = UCDGTBCSAC(config, device)

    src_buffer = ReplayBuffer(
        state_dim=5,
        action_dim=2,
        device=device,
        max_size=1000,
    )

    tar_buffer = ReplayBuffer(
        state_dim=5,
        action_dim=2,
        device=device,
        max_size=1000,
    )

    def add_transition(buffer, shift=0.0):
        state = np.random.randn(5).astype(np.float32)
        action = np.random.uniform(
            -1, 1, size=2
        ).astype(np.float32)

        delta = np.zeros(5, dtype=np.float32)
        delta[:2] = action
        delta += shift

        next_state = state + delta
        reward = -float(np.sum(next_state ** 2))

        buffer.add(
            state,
            action,
            next_state,
            reward,
            0.0,
        )

    # Populate source data.
    for _ in range(200):
        add_transition(src_buffer, shift=0.2)

    # Populate fewer than 64 target transitions.
    for _ in range(32):
        add_transition(tar_buffer, shift=0.0)

    print("Testing target-only warm-up...")

    policy.train(
        src_buffer,
        tar_buffer,
        batch_size=16,
    )

    assert not policy.ucdgt_ready
    print("PASS: Target-only warm-up")

    # Reach the calibration threshold.
    for _ in range(32):
        add_transition(tar_buffer, shift=0.0)

    print("Testing dynamics initialization...")

    policy.train(
        src_buffer,
        tar_buffer,
        batch_size=16,
    )

    assert policy.ucdgt_ready
    assert policy.ucdgt.calibrated

    print("PASS: Dynamics calibration")

    src_state, src_action, src_next_state, _, _ = (
        src_buffer.sample(16)
    )

    weights = policy._source_weights(
        src_state,
        src_action,
        src_next_state,
    )

    assert weights.shape == (16, 1)
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    assert (weights <= 1).all()

    print("PASS: Source compatibility weights")
    print("Mean source weight:", weights.mean().item())

    print("Testing weighted training update...")

    policy.train(
        src_buffer,
        tar_buffer,
        batch_size=16,
    )

    for parameter in policy.policy.parameters():
        assert torch.isfinite(parameter).all()

    for parameter in policy.q_funcs.parameters():
        assert torch.isfinite(parameter).all()

    print("PASS: Weighted training update")
    print("PASS: All UCDGT-BC-SAC integration tests")


if __name__ == "__main__":
    main()
