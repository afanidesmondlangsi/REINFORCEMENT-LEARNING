
import torch


def weighted_bc_loss(
    predicted_actions,
    source_actions,
    weights,
    eps=1e-8,
):
    """
    Compatibility-weighted behavioral cloning loss.

    predicted_actions: [B, action_dim]
    source_actions:    [B, action_dim]
    weights:           [B] or [B, 1]

    Returns the weighted mean squared error.

    When every weight equals one, this reduces
    to the original torch.nn.functional.mse_loss.
    """
    if predicted_actions.shape != source_actions.shape:
        raise ValueError("Action tensor shapes must match.")

    weights = weights.reshape(-1)

    if weights.shape[0] != predicted_actions.shape[0]:
        raise ValueError("Weights must match batch size.")

    if not torch.isfinite(weights).all():
        raise ValueError("Compatibility weights must be finite.")

    if torch.any(weights < 0):
        raise ValueError("Compatibility weights must be nonnegative.")

    # Mean squared error over action dimensions.
    per_transition_loss = (
        predicted_actions - source_actions
    ).pow(2).mean(dim=-1)

    # Preserve original MSE when weights are uniform.
    weight_sum = weights.sum()

    if weight_sum.item() <= eps:
        # No source BC gradient when all transitions are rejected.
        return per_transition_loss.sum() * 0.0

    return (
        weights * per_transition_loss
    ).sum() / weight_sum
