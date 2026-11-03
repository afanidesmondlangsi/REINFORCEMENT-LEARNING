
import torch
from unittest.mock import patch

from algo.offline_online.ucdgt_bc_sac import UCDGTBCSAC


def main():
    torch.manual_seed(42)

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
        "ucdgt_ensemble_size": 3,
        "ucdgt_gate_tau": 10.0,
    }

    policy = UCDGTBCSAC(
        config,
        torch.device("cpu"),
    )

    policy.ucdgt.mismatch_reference = torch.tensor(1.0)
    policy.ucdgt.uncertainty_reference = torch.tensor(1.0)

    state = torch.zeros(4, 5)
    action = torch.zeros(4, 2)
    next_state = torch.zeros(4, 5)

    mismatch = torch.tensor([2.0, 2.0, 2.0, 2.0])
    uncertainty = torch.tensor([0.5, 1.0, 2.0, 5.0])

    with patch.object(
        policy.ucdgt,
        "calculate_statistics",
        return_value=(mismatch, uncertainty),
    ):
        weights = policy._source_weights(
            state,
            action,
            next_state,
        )

    expected = torch.tensor([
        [0.904837],
        [0.904837],
        [0.818731],
        [0.606531],
    ])

    print("Actual weights:")
    print(weights)

    assert weights.shape == (4, 1)

    assert torch.allclose(
        weights,
        expected,
        atol=1e-5,
    )

    assert torch.all(
        weights[1:] <= weights[:-1]
    )

    print("PASS: Actual UCDGT gate matches expected weights")
    print("PASS: Uncertainty monotonicity")
    print("PASS: UCDGT gate method test")


if __name__ == "__main__":
    main()
