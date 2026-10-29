
import torch
import torch.nn.functional as F

from algo.offline_online.ea_ntc.compatibility import weighted_bc_loss


def test_uniform_weights():
    torch.manual_seed(42)

    predicted = torch.randn(128, 3, requires_grad=True)
    target = torch.randn(128, 3)
    weights = torch.ones(128)

    original = F.mse_loss(predicted, target)
    weighted = weighted_bc_loss(predicted, target, weights)

    assert torch.allclose(original, weighted, atol=1e-7)

    grad_original = torch.autograd.grad(
        original, predicted, retain_graph=True
    )[0]

    grad_weighted = torch.autograd.grad(
        weighted, predicted
    )[0]

    assert torch.allclose(
        grad_original, grad_weighted, atol=1e-7
    )

    print("PASS: Original BC-SAC loss and gradients match.")


def test_rejected_transition():
    predicted = torch.tensor(
        [[1.0], [3.0]], requires_grad=True
    )
    target = torch.zeros_like(predicted)
    weights = torch.tensor([1.0, 0.0])

    loss = weighted_bc_loss(predicted, target, weights)

    assert torch.allclose(loss, torch.tensor(1.0))
    print("PASS: Incompatible transition excluded.")


def test_all_rejected():
    predicted = torch.randn(4, 3, requires_grad=True)
    target = torch.randn(4, 3)
    weights = torch.zeros(4)

    loss = weighted_bc_loss(predicted, target, weights)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.all(predicted.grad == 0)

    print("PASS: All-rejected batch handled safely.")


if __name__ == "__main__":
    test_uniform_weights()
    test_rejected_transition()
    test_all_rejected()
