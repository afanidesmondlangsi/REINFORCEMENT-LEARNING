"""Run from the repository root: python test_ucdgt_periodic_refresh.py"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'algo'))
from offline_online.ucdgt_bc_sac import UCDGTBCSAC

class FakeBuffer:
    def __init__(self, capacity=8):
        self.max_size = capacity
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((capacity, 2), dtype=np.float32)
        self.action = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, 2), dtype=np.float32)
        self.insertions = 0

    def add(self):
        self.state[self.ptr, :] = self.insertions
        self.action[self.ptr, :] = self.insertions
        self.next_state[self.ptr, :] = self.insertions + 1
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.insertions += 1

class FakeCore:
    def __init__(self):
        self.seen = []
        self.calibration_count = 0
        self.calibrated = True

    def train_dynamics(self, state, action, next_state):
        assert self.calibrated or self.calibration_count >= 1
        self.seen.extend(state[:, 0].cpu().numpy().astype(int).tolist())
        self.calibrated = False

    def calibrate(self, state, action, next_state):
        assert state[:, 0].tolist() == [1.0, 3.0]
        self.calibration_count += 1
        self.calibrated = True

buffer = FakeBuffer()
for _ in range(5):
    buffer.add()

agent = UCDGTBCSAC.__new__(UCDGTBCSAC)
agent.ucdgt_ready = True
agent.ucdgt_update_interval = 2
agent.ucdgt_update_steps = 1
agent.ucdgt_update_window = 8
agent.ucdgt_last_update_size = 5
agent.ucdgt_target_insertions = 5
agent.ucdgt_last_observed_ptr = buffer.ptr
agent.ucdgt_last_observed_size = buffer.size
agent.ucdgt_calibration_indices = np.array([1, 3])
agent.ucdgt_calibration_data = (
    torch.tensor([[1., 1.], [3., 3.]]),
    torch.tensor([[1.], [3.]]),
    torch.tensor([[2., 2.], [4., 4.]]),
)
agent.ucdgt_dynamics_updates = 0
agent.ucdgt_refresh_count = 0
agent.ucdgt = FakeCore()
agent.device = torch.device('cpu')
agent.total_it = 0

assert agent._refresh_ucdgt(buffer) is False
buffer.add()
assert agent._refresh_ucdgt(buffer) is False
buffer.add()
assert agent._refresh_ucdgt(buffer) is True
assert agent.ucdgt_target_insertions == 7
assert set(agent.ucdgt.seen).isdisjoint({1, 3})
assert agent.ucdgt.calibrated
print('PASS: refresh interval and calibration exclusion before wraparound')

buffer.add()  # 8, full
assert agent._refresh_ucdgt(buffer) is False
buffer.add()  # 9, overwrite original index 0
assert agent._refresh_ucdgt(buffer) is True
assert agent.ucdgt_target_insertions == 9
assert set(agent.ucdgt.seen).isdisjoint({1, 3})
print('PASS: refresh scheduling after buffer becomes full')

buffer.add()  # 10, overwrite original calibration index 1
assert agent._refresh_ucdgt(buffer) is False
buffer.add()  # 11, overwrite original index 2
assert agent._refresh_ucdgt(buffer) is True
assert agent.ucdgt_target_insertions == 11
assert 9 in agent.ucdgt.seen  # overwritten calibration slot is reusable
assert 3 not in agent.ucdgt.seen  # original index 3 not overwritten yet
print('PASS: transition-identity exclusion and overwritten slot reuse')

buffer.add()  # 12, overwrite original calibration index 3
assert agent._refresh_ucdgt(buffer) is False
buffer.add()  # 13
assert agent._refresh_ucdgt(buffer) is True
assert agent.ucdgt_target_insertions == 13
assert agent.ucdgt_refresh_count == 4
assert agent.ucdgt_dynamics_updates == 4
assert agent.ucdgt.calibration_count == 4
assert agent._refresh_ucdgt(buffer) is False
print('PASS: calibration restoration, counters, and no duplicate refresh')
print('PASS: all periodic refresh tests')
