import sys
import math
import yaml

sys.path.insert(0, "algo")

from offline_online.ea_ntc_bc_sac import EANTCBCSAC

with open(
    "config/mujoco/ea_ntc_bc_sac/halfcheetah.yaml",
    encoding="utf-8",
) as handle:
    config = yaml.safe_load(handle)

assert config.get("ea_transfer_quality_enabled", False)
assert config.get("ea_adaptive_bc_enabled", False), (
    "Add ea_adaptive_bc_enabled: true to the YAML configuration"
)

model = EANTCBCSAC(config, "cpu")

assert model.ea_transfer_quality_enabled
assert model.ea_adaptive_bc_enabled
assert model.ea_bc_coefficient == 0.0

print("Initial beta:", model.ea_bc_coefficient)

positive_beta = model.ea_bc_controller.update(
    quality=1e-3,
    uncertainty=1e-4,
)

negative_beta = model.ea_bc_controller.update(
    quality=-1e-3,
    uncertainty=1e-4,
)

print("Positive-transfer beta:", positive_beta)
print("Negative-transfer beta:", negative_beta)

assert math.isclose(positive_beta, 0.8, abs_tol=1e-8)
assert negative_beta == 0.0

print("PHASE 18 INTEGRATION CHECK PASSED")
