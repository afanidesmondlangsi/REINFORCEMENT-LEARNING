import sys
import torch
import yaml

sys.path.insert(0, "algo")

from offline_online.ea_ntc_bc_sac import EANTCBCSAC

with open(
    "config/mujoco/ea_ntc_bc_sac/halfcheetah.yaml",
    encoding="utf-8",
) as f:
    config = yaml.safe_load(f)

torch.manual_seed(2026)

model = EANTCBCSAC(config, "cpu")

state = torch.randn(128, config["state_dim"])
src_state = torch.randn(64, config["state_dim"])
src_action = torch.tanh(
    torch.randn(64, config["action_dim"])
)
weights = torch.linspace(0.2, 1.0, 64)

parameters = list(model.policy.parameters())


def get_gradient(beta):
    model.ea_bc_coefficient = beta

    model.policy.zero_grad(set_to_none=True)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12345)

        loss, _ = model.update_policy_and_temp(
            state,
            src_state,
            src_action,
            source_weights=weights,
        )

        gradients = torch.autograd.grad(
            loss,
            parameters,
            allow_unused=False,
        )

    return loss.item(), torch.cat(
        [g.detach().reshape(-1) for g in gradients]
    )


loss0, grad0 = get_gradient(0.0)
loss1, grad1 = get_gradient(1.0)
losshalf, gradhalf = get_gradient(0.5)

expected_half = 0.5 * (grad0 + grad1)

gradient_error = (
    gradhalf - expected_half
).abs().max().item()

expected_loss_half = 0.5 * (loss0 + loss1)
loss_error = abs(losshalf - expected_loss_half)

print("Loss beta=0:", loss0)
print("Loss beta=0.5:", losshalf)
print("Loss beta=1:", loss1)

print("Maximum gradient interpolation error:", gradient_error)
print("Loss interpolation error:", loss_error)

assert torch.allclose(
    gradhalf,
    expected_half,
    rtol=1e-4,
    atol=1e-5,
), "Adaptive BC gradient scaling failed"

assert loss_error < 1e-4, (
    "Adaptive BC loss scaling failed"
)

print("PHASE 18 ACTOR-GRADIENT TEST PASSED")
