
import torch

from algo.offline_online.ucdgt_bc_sac import UCDGTBCSAC


def main():
    torch.manual_seed(42)

    device = torch.device("cpu")

    config = {
        "state_dim": 5,
        "action_dim": 2,
        "max_action": 1.0,
        "hidden_sizes": 32,
        "gamma": 0.99,
        "tau": 0.005,
        "update_interval": 2,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "temperature_opt": False,
        "weight": 2.5,
        "ucdgt_warmup": 64,
        "ucdgt_train_steps": 10,
        "ucdgt_cal_fraction": 0.2,
        "ucdgt_ensemble_size": 3,
        "ucdgt_gate_tau": 10.0,
        "ucdgt_source_scale": 1.0,
    }

    policy = UCDGTBCSAC(config, device)

    batch_size = 16

    def make_batch():
        state = torch.randn(batch_size, 5)
        action = torch.rand(batch_size, 2) * 2 - 1
        next_state = state + 0.1 * torch.randn(batch_size, 5)
        reward = torch.randn(batch_size, 1)
        not_done = torch.ones(batch_size, 1)

        return (
            state,
            action,
            next_state,
            reward,
            not_done,
        )

    src = make_batch()
    tar = make_batch()

    src_weights = torch.ones(batch_size, 1)

    # Both functions sample next actions from the
    # stochastic SAC policy. Reset the RNG to make
    # their Bellman targets directly comparable.
    rng_state = torch.get_rng_state()

    original_loss = policy.update_q_functions(
        torch.cat([src[0], tar[0]], dim=0),
        torch.cat([src[1], tar[1]], dim=0),
        torch.cat([src[3], tar[3]], dim=0),
        torch.cat([src[2], tar[2]], dim=0),
        torch.cat([src[4], tar[4]], dim=0),
    )

    torch.set_rng_state(rng_state)

    weighted_loss = policy._update_weighted_q(
        *src,
        *tar,
        src_weights,
    )

    difference = abs(
        original_loss.item() - weighted_loss.item()
    )

    print("Original BC-SAC critic loss:", original_loss.item())
    print("UCDGT critic loss (weights=1):", weighted_loss.item())
    print("Absolute difference:", difference)

    assert torch.allclose(
        original_loss,
        weighted_loss,
        rtol=1e-5,
        atol=1e-6,
    )

    print("PASS: Weighted critic matches BC-SAC when weights=1")


if __name__ == "__main__":
    main()
