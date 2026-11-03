import sys
import math

sys.path.insert(0, "algo")

from offline_online.ea_ntc.adaptive_bc import AdaptiveBCController

controller = AdaptiveBCController(
    confidence_multiplier=2.0,
    uncertainty_scale=2.0,
    max_coefficient=1.0,
)

cases = [
    ("Strong positive", 1.0e-3, 1.0e-4, 0.8),
    ("Weak positive", 2.1e-4, 1.0e-4, 1.0 / 21.0),
    ("Neutral", 0.0, 1.0e-4, 0.0),
    ("Negative", -1.0e-3, 1.0e-4, 0.0),
    ("Uncertain positive", 1.0e-3, 8.0e-4, 0.0),
]

for name, quality, uncertainty, expected in cases:
    beta = controller.update(quality, uncertainty)

    print(
        f"{name:20s} "
        f"quality={quality:+.6e} "
        f"uncertainty={uncertainty:.6e} "
        f"beta={beta:.6f}"
    )

    assert math.isfinite(beta)
    assert 0.0 <= beta <= 1.0
    assert math.isclose(beta, expected, abs_tol=1e-8)

assert controller.update(float("nan"), 1e-4) == 0.0
assert controller.update(1e-3, float("inf")) == 0.0
assert controller.update(1e-3, -1e-4) == 0.0

print("ADAPTIVE BC CONTROLLER TEST PASSED")
