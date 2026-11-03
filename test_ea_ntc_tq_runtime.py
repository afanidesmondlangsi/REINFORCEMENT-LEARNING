import sys
import copy
import torch
import numpy as np

sys.path.insert(0, "algo")

from offline_online.ea_ntc_bc_sac import EANTCBCSAC

config = {
    "state_dim": 17,
    "action_dim": 3,
    "max_action": 1.0,
    "hidden_sizes": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "update_interval": 2,
    "temperature_opt": False,
    "critic_lr": 0.0003,
    "actor_lr": 0.0003,
    "weight": 2.5,
    "ea_gate_bc": True,
    "ea_transfer_quality_enabled": True,
    "ea_tq_interval": 100,
    "ea_tq_batch_size": 64,
    "ea_reliability_mode": "relative",
}

model = EANTCBCSAC(config, "cpu")

class TestReplay:
    def __init__(self):
        self.size = 256
        self.max_size = 10000
        self.state = np.random.randn(256, 17).astype("float32")
        self.action = np.random.randn(256, 3).astype("float32")
        self.next_state = np.random.randn(256, 17).astype("float32")

replay = TestReplay()

model.ea_gate_ready = True
model.ea_last_relative_error = 0.4

src_state = torch.randn(128, 17)
src_action = torch.tanh(torch.randn(128, 3))
weights = torch.ones(128)

before_actor = copy.deepcopy(model.policy.state_dict())
before_optimizer = copy.deepcopy(
    model.policy_optimizer.state_dict()
)
before_rng = torch.get_rng_state().clone()

for p in model.q_funcs.parameters():
    p.requires_grad = False

model._transfer_quality_probe(
    replay,
    src_state,
    src_action,
    weights,
)

for p in model.q_funcs.parameters():
    p.requires_grad = True

actor_unchanged = all(
    torch.equal(value, before_actor[key])
    for key, value in model.policy.state_dict().items()
)

rng_unchanged = torch.equal(
    before_rng, torch.get_rng_state()
)

optimizer_unchanged = (
    str(before_optimizer)
    == str(model.policy_optimizer.state_dict())
)

print("Actor unchanged:", actor_unchanged)
print("Optimizer unchanged:", optimizer_unchanged)
print("PyTorch RNG unchanged:", rng_unchanged)
print("Transfer quality:", model.ea_tq_quality)
print("Transfer uncertainty:", model.ea_tq_uncertainty)
print("Adaptive strength:", model.ea_tq_strength)

assert actor_unchanged
assert optimizer_unchanged
assert rng_unchanged

print("TRANSFER-QUALITY PROBE INTEGRITY TEST PASSED")
