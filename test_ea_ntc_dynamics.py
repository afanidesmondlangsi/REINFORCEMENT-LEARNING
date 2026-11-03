
import torch

from algo.offline_online.ea_ntc.dynamics import (
    TargetDynamicsEnsemble,
)


def test_dynamics_ensemble():
    torch.manual_seed(42)

    state_dim = 17
    action_dim = 3
    batch_size = 128

    ensemble = TargetDynamicsEnsemble(
        state_dim=state_dim,
        action_dim=action_dim,
        device="cpu",
        ensemble_size=3,
    )

    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)

    # Synthetic target dynamics with a learnable mapping.
    delta = torch.zeros_like(states)
    delta[:, :3] = 0.2 * actions
    delta[:, 3:] = 0.1 * states[:, 3:]

    next_states = states + delta

    with torch.no_grad():
        initial_mismatch, _ = ensemble.mismatch(
            states, actions, next_states
        )

    for step in range(200):
        loss = ensemble.train_step(
            states,
            actions,
            next_states,
        )

    final_mismatch, uncertainty = ensemble.mismatch(
        states,
        actions,
        next_states,
    )

    assert final_mismatch.shape == (batch_size,)
    assert uncertainty.shape == (batch_size,)

    assert torch.isfinite(final_mismatch).all()
    assert torch.isfinite(uncertainty).all()

    assert (final_mismatch >= 0).all()
    assert (uncertainty >= 0).all()

    assert final_mismatch.mean() < initial_mismatch.mean()

    print("PASS: Dynamics ensemble trains successfully.")
    print("PASS: Prediction mismatch decreases.")
    print("PASS: Ensemble uncertainty is finite.")

    print(
        "Initial mismatch:",
        initial_mismatch.mean().item()
    )
    print(
        "Final mismatch:",
        final_mismatch.mean().item()
    )
    print(
        "Final training loss:",
        loss
    )


if __name__ == "__main__":
    test_dynamics_ensemble()
