
import torch
import numpy as np

from algo.offline_online.ucdgt_core import UCDGTCore


torch.manual_seed(42)
np.random.seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

state_dim = 5
action_dim = 2
n_samples = 1200

state = torch.randn(
    n_samples, state_dim, device=device
)
action = torch.randn(
    n_samples, action_dim, device=device
)

delta = torch.zeros_like(state)
delta[:, 0] = 0.5 * state[:, 0] + action[:, 0]
delta[:, 1] = -0.3 * state[:, 1] + action[:, 1]
delta[:, 2] = 0.2 * state[:, 2]
delta[:, 3] = -0.1 * state[:, 3]
delta[:, 4] = 0.4 * state[:, 4]

next_state = state + delta

# Disjoint target training and calibration sets.
train_indices = torch.arange(0, 1000, device=device)
cal_indices = torch.arange(1000, 1200, device=device)

ucdgt = UCDGTCore(
    state_dim=state_dim,
    action_dim=action_dim,
    device=device,
    ensemble_size=3,
    learning_rate=1e-3,
    tau=10.0,
)

ucdgt.fit_normalization(
    state[train_indices],
    action[train_indices],
    next_state[train_indices],
)

print("\nNormalization fitted")
print("Delta scale:", ucdgt.delta_scale.cpu().numpy())

print("\nTraining ensemble...")

for step in range(300):
    sampled = train_indices[
        torch.randint(
            0, len(train_indices),
            (128,),
            device=device,
        )
    ]

    loss = ucdgt.train_dynamics(
        state[sampled],
        action[sampled],
        next_state[sampled],
    )

    if (step + 1) % 50 == 0:
        print(
            f"Step {step + 1}: "
            f"loss = {loss:.6f}"
        )

ucdgt.calibrate(
    state[cal_indices],
    action[cal_indices],
    next_state[cal_indices],
)

print("\nHeld-out calibration complete")
print(
    "Mismatch reference:",
    ucdgt.mismatch_reference.item(),
)
print(
    "Uncertainty reference:",
    ucdgt.uncertainty_reference.item(),
)

print("\nDynamics-shift sensitivity")
print(f"{'Shift':<12}{'Mean weight':<15}")

means = []

for shift in [0.0, 0.01, 0.05, 0.1, 0.5]:
    weights = ucdgt.calculate_weights(
        state[cal_indices],
        action[cal_indices],
        next_state[cal_indices] + shift,
    )

    assert torch.isfinite(weights).all()
    assert ((weights >= 0) & (weights <= 1)).all()

    mean_weight = weights.mean().item()
    means.append(mean_weight)

    print(f"{shift:<12.2f}{mean_weight:<15.6f}")

assert means[-1] < means[0], (
    "Large dynamics shift was not downweighted"
)

print("\nPASS: Improved UCDGT core test")
