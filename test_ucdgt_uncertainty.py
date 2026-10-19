
import torch

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

    # Fix mismatch and increase uncertainty.
    mismatch = torch.tensor([2.0, 2.0, 2.0, 2.0])
    uncertainty = torch.tensor([0.5, 1.0, 2.0, 5.0])

    relative_mismatch = (
        mismatch / policy.ucdgt.mismatch_reference
    )

    relative_uncertainty = (
        uncertainty / policy.ucdgt.uncertainty_reference
    )

    penalty = (
        torch.relu(relative_mismatch - 1.0)
        + torch.relu(relative_uncertainty - 1.0)
    ) / policy.ucdgt.tau

    weights = torch.exp(-penalty)

    print("Uncertainty | Compatibility weight")

    for u, w in zip(uncertainty, weights):
        print(f"{u.item():11.2f} | {w.item():.6f}")

    assert torch.all(
        weights[1:] <= weights[:-1]
    )

    print(
        "PASS: Compatibility weights do not increase "
        "with uncertainty"
    )

    # Increasing mismatch should also not increase trust.
    mismatch_levels = torch.tensor(
        [0.5, 1.0, 2.0, 5.0]
    )

    fixed_uncertainty = torch.ones(4)

    mismatch_penalty = torch.relu(
        mismatch_levels
        / policy.ucdgt.mismatch_reference
        - 1.0
    )

    uncertainty_penalty = torch.relu(
        fixed_uncertainty
        / policy.ucdgt.uncertainty_reference
        - 1.0
    )

    mismatch_weights = torch.exp(
        -(
            mismatch_penalty
            + uncertainty_penalty
        ) / policy.ucdgt.tau
    )

    assert torch.all(
        mismatch_weights[1:]
        <= mismatch_weights[:-1]
    )

    print(
        "PASS: Compatibility weights do not increase "
        "with dynamics mismatch"
    )

    print("PASS: Conservative gate mathematical tests")


if __name__ == "__main__":
    main()
