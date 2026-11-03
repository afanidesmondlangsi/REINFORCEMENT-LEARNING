
import torch

from algo.offline_online.ea_ntc.calibration import (
    TargetErrorCalibrator,
)


def test_calibration():
    calibrator = TargetErrorCalibrator(
        gate_temperature=1.0
    )

    # Synthetic held-out target prediction errors.
    target_errors = torch.tensor([
        0.08, 0.09, 0.10, 0.11, 0.12
    ])

    reference = calibrator.fit(target_errors)

    assert abs(reference - 0.10) < 1e-6

    source_errors = torch.tensor([
        0.05, 0.10, 0.20, 0.50
    ])

    weights = calibrator.weights(source_errors)

    assert weights.shape == (4,)
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0)
    assert torch.all(weights <= 1)

    # Errors below target reference retain full weight.
    assert torch.allclose(
        weights[0], torch.tensor(1.0)
    )

    # Error equal to target reference retains full weight.
    assert torch.allclose(
        weights[1], torch.tensor(1.0)
    )

    # Increasing mismatch reduces compatibility.
    assert weights[2] < weights[1]
    assert weights[3] < weights[2]

    print("PASS: Target calibration reference correct.")
    print("PASS: Compatibility weights are bounded.")
    print("PASS: Larger mismatch receives lower weight.")

    print("Target reference:", reference)
    print("Source weights:", weights.tolist())


if __name__ == "__main__":
    test_calibration()
