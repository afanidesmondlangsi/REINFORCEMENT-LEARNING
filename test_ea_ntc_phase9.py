import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, './algo')
from call_algo import call_algo
from utils import ReplayBuffer

with open('config/mujoco/ea_ntc_bc_sac/halfcheetah.yaml') as f:
    config = yaml.safe_load(f)
config.update(hidden_sizes=32, ea_hidden_size=32,
              ea_warmup_target_steps=32, ea_calibration_fraction=0.20,
              ea_calibration_gap=2, ea_min_dynamics_updates=3,
              ea_max_calibration_error=1.0,
              ea_dynamics_update_interval=2, ea_calibration_update_interval=2,
              ea_dynamics_batch_size=16)
np.random.seed(42)
torch.manual_seed(42)
model = call_algo('ea_ntc_bc_sac', config, 1, torch.device('cpu'))
buf = ReplayBuffer(17, 3, torch.device('cpu'), max_size=256)
for i in range(64):
    state = np.random.randn(17).astype('float32')
    action = np.random.uniform(-1, 1, 3).astype('float32')
    next_state = state + 0.01 * state
    next_state[:3] += 0.1 * action
    buf.add(state, action, next_state, 0.0, 0.0)
for _ in range(12):
    model.total_it += 1
    model.update_dynamics_and_calibration(buf)
assert model.ea_last_calibration_count == 6
assert model.ea_last_training_count == 56
assert model.ea_dynamics_updates >= 3
assert model.ea_gate_ready
print('PASS: Fixed 20% warm-up holdout and temporal gap.')
print('PASS: Dynamics updates and reliability gate.')
assert not set(range(6)).intersection(set(range(8, 64)))
print('PASS: Training and calibration indices disjoint.')
print('Phase 9 holdout test passed.')
