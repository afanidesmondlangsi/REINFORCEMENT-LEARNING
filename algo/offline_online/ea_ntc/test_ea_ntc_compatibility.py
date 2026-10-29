
import torch
import torch.nn.functional as F

from algo.offline_online.ea_ntc.compatibility import (
    weighted_bc_loss,
)


def test_uniform_weights():
    torch.manual_seed(42)

    predicted = torch.randn(
        128, 3, requires_grad=True
    )
    target = torch.randn(128, 3)

    weights = torch.ones(128)

    original_loss = F.mse_loss(predicted, target)

    weighted_loss = weighted_bc_loss(
        predicted,
        target,
        weights,
    )

    assert torch.allclose(
        original_loss,
        weighted_loss,
        atol=1e-7,
    )

    original_gradient = torch.autograd.grad(
        original_loss,
        predicted,
        retain_graph=True,
    )[0]

    weighted_gradient = torch.autograd.grad(
        weighted_loss,
        predicted,
    )[0]

    assert torch.allclose(
        original_gradient,
        weighted_gradient,
        atol=1e-7,
    )

    print(
        "PASS: Uniform weights reproduce "
        "original BC-SAC loss and gradients."
    )


def test_rejected_transitions():
    predicted = torch.tensor(
        [[1.0], [3.0]],
        requires_grad=True,
    )

    target = torch.zeros_like(predicted)

    weights = torch.tensor([1.0, 0.0])

    loss = weighted_bc_loss(
        predicted,
        target,
        weights,
    )

    assert torch.allclose(
        loss,
        torch.tensor(1.0),
    )

    print("PASS: Rejected transitions are excluded.")


def test_all_rejected():
    predicted = torch.randn(
        4, 3, requires_grad=True
    )

    target = torch.randn(4, 3)
    weights = torch.zeros(4)

    loss = weighted_bc_loss(
        predicted,
        target,
        weights,
    )

    loss.backward()

    assert loss.item() == 0.0
    assert torch.all(predicted.grad == 0)

    print("PASS: All-rejected batch is handled safely.")


if __name__ == "__main__":
    test_uniform_weights()
    test_rejected_transitions()
    test_all_rejected()
