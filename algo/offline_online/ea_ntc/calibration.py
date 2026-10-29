
import torch


class TargetErrorCalibrator:
    """
    Calibrates source mismatch against held-out
    target transition prediction errors.
    """

    def __init__(
        self,
        gate_temperature=1.0,
        eps=1e-8,
        min_reference=1e-6,
    ):
        if gate_temperature <= 0:
            raise ValueError(
                "Gate temperature must be positive."
            )

        self.gate_temperature = gate_temperature
        self.eps = eps
        self.min_reference = min_reference
        self.target_reference = None

    @torch.no_grad()
    def fit(self, target_errors):
        errors = target_errors.reshape(-1)

        if errors.numel() == 0:
            raise ValueError(
                "Target calibration errors are empty."
            )

        if not torch.isfinite(errors).all():
            raise ValueError(
                "Target errors must be finite."
            )

        if torch.any(errors < 0):
            raise ValueError(
                "Target errors must be nonnegative."
            )

        reference = torch.median(errors)

        self.target_reference = max(
            reference.item(),
            self.min_reference,
        )

        return self.target_reference

    @torch.no_grad()
    def weights(self, source_errors):
        if self.target_reference is None:
            raise RuntimeError(
                "Calibrator must be fitted first."
            )

        errors = source_errors.reshape(-1)

        if not torch.isfinite(errors).all():
            raise ValueError(
                "Source errors must be finite."
            )

        if torch.any(errors < 0):
            raise ValueError(
                "Source errors must be nonnegative."
            )

        ratio = errors / (
            self.target_reference + self.eps
        )

        excess_mismatch = torch.clamp(
            ratio - 1.0,
            min=0.0,
        )

        weights = torch.exp(
            -excess_mismatch / self.gate_temperature
        )

        return weights
