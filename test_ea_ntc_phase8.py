"""Run from repository root after replacing ea_ntc_bc_sac.py."""
import sys
import torch
sys.path.insert(0, './algo')
from offline_online.ea_ntc_bc_sac import EANTCBCSAC

class FakeDynamics:
    def mismatch(self, state, action, next_state):
        return torch.tensor([0.1, 0.1, 0.1]), torch.tensor([0.1, 0.2, 0.8])

class FakeCalibrator:
    def weights(self, errors):
        return torch.ones_like(errors) * 0.8

def check():
    p = object.__new__(EANTCBCSAC)
    p.ea_enabled = True
    p.ea_gate_ready = True
    p.dynamics = FakeDynamics()
    p.calibrator = FakeCalibrator()
    p.ea_uncertainty_enabled = False
    p.ea_uncertainty_temperature = 1.0
    p.ea_target_uncertainty_reference = 0.2
    p.ea_last_uncertainty = None
    p.ea_last_uncertainty_penalty = None
    z = torch.zeros(3, 1)
    mismatch_only = p.source_weights(z, z, z)
    assert torch.allclose(mismatch_only, torch.full((3,), 0.8))
    p.ea_uncertainty_enabled = True
    combined = p.source_weights(z, z, z)
    assert torch.isfinite(combined).all()
    assert torch.all((combined >= 0) & (combined <= mismatch_only))
    assert combined[0] == combined[1]
    assert combined[2] < combined[1]
    assert torch.allclose(combined[2], 0.8 * torch.exp(torch.tensor(-3.0)))
    p.ea_target_uncertainty_reference = None
    assert torch.all(p.source_weights(z, z, z) == 0)
    p.ea_enabled = False
    assert torch.all(p.source_weights(z, z, z) == 1)
    print('PASS: Mismatch-only gate preserved.')
    print('PASS: Uncertainty penalty bounded and monotonic.')
    print('PASS: Uncalibrated uncertainty abstains.')
    print('PASS: Disabled gate gives unit weights.')

if __name__ == '__main__':
    check()
